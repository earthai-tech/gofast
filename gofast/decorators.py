# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio~@Daniel03 <etanoyau@gmail.com>
"""
decorators
=============

This module provides a collection of decorators designed to enhance and simplify 
common programming tasks in Python. These decorators offer functionality ranging 
from suppressing output and sanitizing docstrings to appending documentation and 
managing feature importance plots. Each decorator is crafted to be reusable and 
easy to integrate into various projects.

Decorators included in this module:
- `SuppressOutput`: Context manager and decorator for suppressing stdout and 
   stderr messages.
- `SanitizeDocstring`: Cleans and restructures a function's or class's docstring 
   to adhere to the Numpy docstring standard.
- `AppendDocFrom`: Appends a specific section of a function's or class's 
   docstring to another.
- `PlotFeatureImportance`: Decorator for plotting permutation feature importance (PFI) 
   diagrams or dendrogram figures.
- `RedirectToNew`: Redirects calls from deprecated functions or classes to their 
   new implementations.
- `SanitizeDocstring`: Sanitizes and restructures docstrings to fit the Numpy
   docstring format.
- More ...

Each decorator is designed with specific use cases in mind, ranging from 
improving code documentation and readability to controlling the output of 
scripts for cleaner execution logs. Users are encouraged to explore the 
functionalities provided by each decorator to enhance their codebase.

Examples:
    >>> from gofast.decorators import SuppressOutput, SanitizeDocstring, AppendDocFrom

Note:
    While each decorator is designed to be as versatile as possible, users should
    consider their specific needs and test decorators in their environment to ensure
    compatibility and desired outcomes.

Contributions:
    - Various contributors and examples from online resources have inspired 
      these decorators.
"""

from __future__ import print_function 
import functools
import inspect
import os
import re
import sys
import warnings
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from ._gofastlog import gofastlog
from ._typing import Union, Optional, Callable
_logger = gofastlog.get_gofast_logger(__name__)

__docformat__='restructuredtext'

class DynamicMethod:
    """
    A class-based decorator designed to preprocess data before it's passed to 
    a function or method. This preprocessing includes filtering data by type, 
    selecting specific columns, handling missing values, applying transformations,
    and executing based on custom conditions. It offers advanced options like 
    treating integer columns as categorical and encoding categorical columns.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the expected data type for processing. The options are:
        - 'numeric': Only numeric columns are considered.
        - 'categorical': Only categorical columns are considered.
        - 'both': Both numeric and categorical columns are considered.
        Defaults to 'numeric'.

    capture_columns : bool, optional
        If set to True, the decorator filters the DataFrame columns to those 
        specified in the 'columns' keyword argument passed to the decorated function.
        Defaults to False.

    treat_int_as_categorical : bool, optional
        When True, integer columns in the DataFrame are treated as categorical data,
        which can be particularly useful for statistical operations that distinguish
        between numeric and categorical data types, such as ANOVA tests.
        Defaults to False.

    encode_categories : bool, optional
        If True, categorical columns are encoded into integer values. This is especially
        useful for models that require numerical input for categorical data.
        Defaults to False.

    drop_na : bool, optional
        Determines whether rows or columns with missing values should be dropped. The
        specific rows or columns to drop are dictated by `na_axis` and `na_thresh`.
        Defaults to False.

    na_axis : Union[int, str], optional
        Specifies the axis along which to drop missing values. Acceptable values are:
        - 0 or 'row': Drop rows with missing values.
        - 1 or 'col': Drop columns with missing values.
        Defaults to 0.

    na_thresh : Optional[float], optional
        Sets a threshold for dropping rows or columns with missing values. This can be
        specified as an absolute number of non-NA values or a proportion (0 < value <= 1)
        of the total number of values in a row or column.
        Defaults to None.

    transform_func : Optional[Callable], optional
        A custom function to apply to the DataFrame before passing it to the decorated
        function. This allows for flexible data transformations as needed.
        Defaults to None.

    condition : Optional[Callable[[pd.DataFrame], bool]], optional
        A condition function that takes the DataFrame as an argument and returns True
        if the decorated function should be executed. This enables conditional processing
        based on the data.
        Defaults to None.

    reset_index : bool, optional
        If True, the DataFrame index is reset before processing. This is useful after
        filtering rows to ensure the index is continuous.
        Defaults to False.

    verbose : bool, optional
        Controls the verbosity of the decoration process. If True, detailed information
        about the preprocessing steps is printed.
        Defaults to False.

    Raises
    ------
    ValueError
        If the first argument to the decorated function is not a pandas DataFrame,
        dictionary, or NumPy ndarray.

    Examples
    --------
    >>> from gofast.decorators import DynamicMethod
    >>> @DynamicMethod(expected_type='numeric', capture_columns=True, 
    ... verbose=True, drop_na=True, na_axis='row', na_thresh=0.5, reset_index=True)
    ... def calculate_variance(data):
    ...     return data.var(ddof=0).mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, np.nan, 6]})
    >>> print(calculate_variance(data))
    # The above example demonstrates preprocessing a DataFrame by dropping rows with
    # more than 50% missing values and calculating the variance of the remaining data.

    Notes
    -----
    - The `treat_int_as_categorical` and `encode_categories` parameters offer flexibility
      in handling integer and categorical data, which can be critical for certain types
      of analysis or modeling.
    - The `transform_func` and `condition` parameters allow for custom data transformations
      and conditional execution, adding a layer of customization to the preprocessing steps.
    """
    def __init__(
        self, 
        expected_type: str = 'numeric', 
        capture_columns: bool = False,
        treat_int_as_categorical: bool = False, 
        encode_categories: bool = False, 
        verbose: bool = False, 
        drop_na: bool = False, 
        na_axis: Union[int, str] = 0, 
        na_thresh: Optional[float] = None, 
        transform_func: Optional[Callable] = None, 
        condition: Optional[Callable[[pd.DataFrame], bool]] = None, 
        reset_index: bool = False
        ):
        self.expected_type = expected_type
        self.capture_columns = capture_columns
        self.treat_int_as_categorical = treat_int_as_categorical
        self.encode_categories = encode_categories
        self.drop_na = drop_na
        self.na_axis = na_axis
        self.na_thresh = na_thresh
        self.transform_func = transform_func
        self.condition = condition
        self.reset_index = reset_index
        self.verbose = verbose

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(f"Preprocessing data for {func.__name__}...")

            data = self._validate_and_prepare_data(args[0], **kwargs)
            if data is None:
                return func(*args, **kwargs)  # Early exit if data validation fails

            data = self._process_data(data, **kwargs)
            return func(data, *args[1:], **kwargs)
        
        self._add_method_to_pandas (wrapper)
        
        return wrapper

    
    def _validate_and_prepare_data(self, data, **kwargs):
        """
        Validates the input data and converts it to a pandas DataFrame if
        necessary.
    
        This method checks if the first argument is one of the supported types 
        (pd.DataFrame, dict, np.ndarray, or iterable object) and converts it to 
        a pandas DataFrame if it's not already one. If `columns` are specified in 
        kwargs and the data is an np.ndarray or a converted iterable, it attempts 
        to use these columns when creating the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame, dict, or np.ndarray
            The input data to be validated and possibly converted.
        **kwargs : dict
            Additional keyword arguments, including 'columns' which may be used 
            if the data is an np.ndarray to specify DataFrame column names.
    
        Returns
        -------
        pd.DataFrame
            The validated and prepared pandas DataFrame.
    
        Raises
        ------
        ValueError
            If the input data is not one of the supported types.
        """
        if isinstance(data, dict):
            data= pd.DataFrame(data)
        # Convert iterable (not DataFrame or np.ndarray) to DataFrame
        elif  hasattr(data, '__iter__') and not isinstance(
                data, ( pd.DataFrame, np.ndarray, pd.Series)):  
            try:
                data =  np.array(data)
            except Exception:
                raise ValueError(
                    "Expect the first argument to be an iterable object"
                    " with minimum samples equal 2.")
        if isinstance(data, np.ndarray):
            columns = kwargs.get('columns')
            if isinstance (columns, str): 
                columns =[columns]
                
            data = pd.DataFrame(data, columns=(
                columns if columns and len(columns) == data.shape[1] else None))
            
        elif isinstance ( data, pd.Series): 
            data = pd.DataFrame ( data)
            
        # Finally validate  whether dataFrame if constructed.
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input data must be a pd.DataFrame, dict, np.ndarray,"
                " or iterable.")
        return data
    
    def _process_data(self, data: pd.DataFrame, **kwargs):
        """
        Applies various preprocessing steps to the data based on the decorator's parameters.
    
        This method sequentially processes the data through specified steps: capturing 
        specified columns, filtering data by type, dropping missing values, applying 
        a transformation function, checking a condition for execution, and resetting 
        the index if required.
    
        Parameters
        ----------
        data : pd.DataFrame
            The data to be processed.
        **kwargs : dict
            Additional keyword arguments passed through from the decorator.
    
        Returns
        -------
        None
            The function modifies the data in place and does not return any value.
        """
        if self.capture_columns:
            data = self._capture_columns(data, **kwargs)
        if self.expected_type in ['numeric', 'categorical']:
            data = self._filter_data_type(data)
        if self.drop_na:
            data = self._drop_na(data)
        if self.transform_func:
            data = self.transform_func(data)
        if self.condition and not self.condition(data):
            if self.verbose:
                print("Condition for execution not met, skipping function call.")
            return
        if self.reset_index:
            data = data.reset_index(drop=True)

        return data 
        
    def _capture_columns(self, data: pd.DataFrame, **kwargs):
        """
        Filters the columns of the DataFrame based on the specified 'columns' in kwargs.
    
        If the 'columns' keyword argument is provided, this method attempts to filter 
        the DataFrame to include only those columns. If any specified columns do not 
        exist in the DataFrame, a warning is printed if verbose output is enabled.

        """
        columns = kwargs.pop('columns', None)
        if columns is not None:
            try:
                data = data[columns]
            except KeyError:
                if self.verbose:
                    print("Specified columns do not match, ignoring columns.")
        return data 
    
    def _filter_data_type(self, data: pd.DataFrame):
        """
        Filters the data based on the expected type ('numeric' or 'categorical').
    
        This method filters the DataFrame to include only numeric or categorical 
        columns based on the `expected_type` parameter. If 'categorical' is specified, 
        it further processes categorical data as per the class parameters.
        """
        if self.expected_type == 'numeric':
            data = data.select_dtypes(include=[np.number])
        elif self.expected_type == 'categorical':
            data =self._handle_categorical_data(data)
        
        return data 
    
    def _handle_categorical_data(self, data: pd.DataFrame):
        """
        Handles categorical data by treating integer columns as categorical
        if specified, and encoding categorical columns if required.
    
        This method processes integer columns as categorical if
        `treat_int_as_categorical` 
        is True, and encodes categorical columns into integers if 
        `encode_categories` is True.
    
        """
        if self.treat_int_as_categorical:
            int_columns = data.select_dtypes(include=[int]).columns.tolist()
            data[int_columns] = data[int_columns].astype('category')
        if self.encode_categories:
            data = self._encode_categorical_columns(data)
        
        return data 
    
    def _encode_categorical_columns(self, data: pd.DataFrame):
        """
        Encodes categorical columns in the DataFrame into integer values.
    
        This method applies Label Encoding to columns in the DataFrame that are 
        identified as categorical (either 'category' or 'object' dtype).

        """
        from sklearn.preprocessing import LabelEncoder
        cat_columns = data.select_dtypes(include=['category', 'object']).columns
        for col in cat_columns:
            data[col] = LabelEncoder().fit_transform(data[col])
            
        return data 

    def _drop_na(self, data: pd.DataFrame):
        """
        Drops rows or columns from the DataFrame based on missing values criteria.
    
        This method drops rows or columns with missing values based on the specified
        `na_axis` and `na_thresh` parameters. `na_axis` determines the axis along which
        to drop (rows or columns), and `na_thresh` specifies the threshold for dropping
        either as an absolute number of non-NA values required to keep a row/column or
        as a proportion of the total number of values.
    
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame from which rows or columns will be dropped based 
            on missing values.
    
        Modifies
        --------
        data : pd.DataFrame
            The input DataFrame is modified in place by dropping specified 
            rows or columns.
    
        """
        # Convert na_axis from string to integer if necessary
        na_axis = 0 if self.na_axis in [0, 'row'] else 1 if self.na_axis in [
            1, 'col'] else self.na_axis
        
        # Calculate the threshold for dropping based on the proportion 
        # if specified as a float less than 1
        if 0 < self.na_thresh <= 1:
            total_elements = len(data.columns) if na_axis == 0 else len(data)
            thresh = int(total_elements * self.na_thresh)
        else:
            thresh = self.na_thresh
        print("na-aixs=", na_axis , "na_thresh=",thresh)
        # Drop missing values based on the specified axis and threshold
        return data.dropna(axis=na_axis, thresh=thresh)
    
    def _add_method_to_pandas(self, func):
        """
        Dynamically adds a new method to pandas DataFrame and Series objects.
        
        This function checks if the method named after `func.__name__` does 
        not already exist in the pandas DataFrame and Series classes. If not,
        it adds `func` as a method to these classes, allowing for seamless 
        integration of custom functionality into pandas objects.
        
        Parameters
        ----------
        func : function
            The function to add as a method to DataFrame and Series objects.
            The name of the function (`func.__name__`) will be used as the 
            method name.
        """
        for cls in [pd.DataFrame, pd.Series]:
            if not hasattr(cls, func.__name__):
                try:
                    setattr(cls, func.__name__, func)
                except Exception as e: # noqa
                    pass 
                    # Optionally log the error or handle it as needed
                    # print(f"Error adding method {func.__name__} to {cls.__name__}: {e}")
class ExportData:
    """
    A decorator for exporting data into various formats post-function execution. 
    It supports exporting pandas DataFrames or other data types to specified 
    file formats with additional customization through keyword arguments.

    Parameters
    ----------
    export_type : str, optional
        The type of data to export, which can be 'frame' for pandas DataFrames or 'text' 
        for text files. Defaults to 'frame'.
    encoding : str, optional
        The encoding to use for text files. Defaults to 'utf-8'. This parameter is not 
        applicable when exporting DataFrames.
    **kwargs : dict
        Additional keyword arguments to be passed to the pandas export function or 
        the file writing process.

    Examples
    --------
    >>> from gofast.decorators import ExportData
    >>> @ExportData(export_type='frame', file_format='csv')
    ... def data_processing_function():
    ...     df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    ...     return df, 'output_filename', 'csv', './savepath', 'Data', {}
    ...
    >>> data_processing_function()
    # This will save the DataFrame returned by data_processing_function to a CSV file
    # named 'output_filename.csv' in the './savepath' directory.
    """
    
    def __init__(self, export_type='frame', encoding='utf8', **kwargs):
        self.export_type = export_type
        self.encoding = encoding
        self.kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_func(*args, **func_kwargs):
            # Extracting data and parameters from the decorated function
            dfs, fname, file_format, savepath, nameof, extra_kwargs = func(
                *args, **func_kwargs)
            
            if self.kwargs.get("file_format") and file_format: 
                # The priority is given to user format. 
                _= self.kwargs.pop("file_format", None)
            
            # Merge decorator's kwargs with function's extra_kwargs,
            # giving precedence to extra_kwargs
            export_kwargs = {**self.kwargs, **extra_kwargs}

            # Ensure the savepath exists
            os.makedirs(savepath, exist_ok=True)

            # Setting file format based on extension or provided format
            _, ext = os.path.splitext(fname)
            file_format = ext.lower() if ext else f".{file_format}"

            # Validate file format
            if self.export_type.lower() == 'frame' and file_format not in ['.csv', '.xlsx']:
                raise ValueError(f"Unsupported file format for DataFrame export: {file_format}")

            # Choose the writer function based on the export type
            if self.export_type.lower() == 'frame':
                fnames = self._export_frame(dfs, fname, file_format, savepath, nameof, **export_kwargs)
            else:
                fnames = self._export_others(dfs, fname, file_format, savepath, nameof, **export_kwargs)

            # Optionally move files to a designated output directory
            # Assuming move_cfile function exists and is imported correctly
            for fname in fnames:
                from .tools.coreutils import move_cfile 
                move_cfile(fname, savepath, dpath='_out')
                
            # Optionally return the filenames of the exported files
            return fnames
        return wrapper_func
        
    def _export_frame(self, dfs, fname, file_format, savepath, nameof=None, **kwargs):
        """
        Handles exporting pandas DataFrame(s) to specified file formats.
        """
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        fnames = []
        for i, df in enumerate(dfs):
            if not isinstance(df, pd.DataFrame):
                continue
            name_suffix = f"_{nameof[i]}" if nameof and i < len(nameof) else ""
            output_fname = f"{fname}{name_suffix}{file_format}"
            full_path = os.path.join(savepath, output_fname)
            
            if file_format == '.xlsx':
                with pd.ExcelWriter(full_path) as writer:
                    df.to_excel(writer, sheet_name=nameof[i] if nameof and i < len(nameof) else 'Sheet1', **kwargs)
            else:
                df.to_csv(full_path, **kwargs)
            fnames.append(full_path)
        
        return fnames

    def _export_others(self, data, fname, file_format, savepath, nameof=None,
                       **kwargs):
        """
        Handles exporting non-DataFrame data to files, primarily text files.
        """
        output_fname = f"{fname}{file_format}"
        full_path = os.path.join(savepath, output_fname)
        
        with open(full_path, mode='w', encoding=self.encoding) as f:
            for item in data:
                f.write(f"{item}\n")
                
        return [full_path]
  
class Temp2D:
    """
    A decorator for creating two-dimensional plots from the outputs of 
    decorated functions. It integrates seamlessly with matplotlib for 
    plotting and supports customization through various parameters.

    Parameters
    ----------
    reason : str, optional
        The purpose of the plot. This parameter is for documentation 
        purposes and does not affect the plot's appearance or behavior.
    **kwargs : dict
        Additional keyword arguments for plot customization. These 
        arguments are expected to align with the parameters used by 
        matplotlib and related plotting utilities.

    Notes
    -----
    The decorator uses the last return value of the decorated function as a 
    dictionary of plotting arguments, which should include keys and values 
    compatible with `matplotlib.pyplot` functions. If these plotting 
    arguments are not provided, an AttributeError will be raised.

    Examples
    --------
    >>> from gofast.decorators import Temp2D
    >>> @Temp2D(reason="Show an example")
    ... def generate_data():
    ...     x = np.linspace(0, 10, 100)
    ...     y = np.sin(x)
    ...     return x, y, {'xlabel': 'X Axis', 'ylabel': 'Y Axis'}
    ...
    >>> generate_data()
    # This will plot a sine wave with the specified x and y labels.
    """

    def __init__(self, reason=None, **kwargs):
        self.reason = reason
        self.plot_kwargs = kwargs

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function and expect a tuple where
            # the last item is a dictionary for plot customization
            *plot_data, plot_customization = func(*args, **kwargs)

            # Update the plot customization with any additional kwargs
            # provided during the decorator initialization
            plot_customization.update(self.plot_kwargs)

            # Call the plot creation method
            return self.plot2d(*plot_data, **plot_customization)

        return wrapper

    def plot2d(self, x, y, **kwargs):
        """
        Generates a 2D plot based on the provided x and y data along with 
        customizable plotting arguments.

        Parameters
        ----------
        x : array-like
            X-coordinates for the plot.
        y : array-like
            Y-coordinates for the plot.
        **kwargs : dict
            Additional keyword arguments for customizing the plot, such as 
            'xlabel', 'ylabel', and any other matplotlib.axes.Axes method 
            arguments.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object with the plot.

        Example
        -------
        >>> Temp2D().plot2d(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)),
        ...                set_xlabel="X Axis", set_ylabel="Y Axis", set_title="Sine Wave")
        # This will create and display a 2D plot of a sine wave.
        """
        fig, ax = plt.subplots()
        ax.plot(x, y)

        # Apply customizations from kwargs
        for key, value in kwargs.items():
            if key in ["ylabel", 'xlabel', 'title']: 
                key = f"set_{key}" # for Axes 
            if hasattr(ax, key) and callable(getattr(ax, key)):
                getattr(ax, key)(value)
            else:
                print(f"Warning: {key} is not a valid Axes method")

        plt.show()
        return ax

    def __getattr__(self, name):
        # Custom error message for missing attributes
        msg = (f"{self.__class__.__name__!r} has no attribute {name!r}. "
               "Ensure plot arguments are supplied as the last return value "
               "of the decorated function.")
        raise AttributeError(msg)

class SignalFutureChange:
    """
    A decorator that signals an upcoming change to a function or class, such 
    as deprecation or a recommendation to use a more robust alternative. It 
    allows the function or class to execute normally while optionally logging 
    a warning message.

    Parameters
    ----------
    message : str, optional
        A message to be displayed to indicate the reason for the future change. 
        This could inform about deprecation or suggest using an alternative.

    Examples
    --------
    >>> from gofast.decorators import SignalFutureChange
    >>> @SignalFutureChange(message="This function will be deprecated in future "
    ...                        "releases. Consider using `new_function` instead.")
    ... def old_function():
    ...     print("This is an old function.")
    ...
    >>> old_function()
    # Executes old_function, logging a message about future deprecation or 
    # recommending an alternative, based on the provided message.
    """
    
    def __init__(self, message=None):
        self.message = message

    def __call__(self, cls_or_func):
        if self.message:
            # Log the warning message at the time of decoration, not at call time
            warnings.warn(self.message, FutureWarning, stacklevel=2)
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        return wrapper

class AppendDocReferences:
    """
    A decorator for appending reStructuredText references to the docstring 
    of the decorated function or class, enhancing Sphinx documentation by 
    auto-retrieving and replacing values from specified references.

    This allows for dynamic insertion of common documentation elements, 
    such as glossary terms or external documentation links, into the 
    docstrings of multiple functions or classes.

    Parameters
    ----------
    docref : str, optional
        The documentation reference string to be appended to the function's 
        or class's docstring. This should be in reStructuredText format.

    Examples
    --------
    >>> from gofast.decorators import AppendDocReferences
    >>> @AppendDocReferences(docref=".. |VES| replace:: Vertical Electrical"
    ...                         " Sounding\\n.. |ERP| replace:: Electrical Resistivity Profiling")
    ... def example_function():
    ...     '''This function demonstrates appending doc references.
    ...
    ...     See more details about |VES| and |ERP|.
    ...     '''
    ...     pass
    ...
    >>> print(example_function.__doc__)
    # The docstring of example_function will now include the replaced 
    # references to VES and ERP along with their definitions.
    """

    def __init__(self, docref=None):
        self.docref = "\n" + docref if docref else ""

    def __call__(self, cls_or_func):
        
        original_doc = cls_or_func.__doc__ if cls_or_func.__doc__ else ''
        # Append the doc reference to the original docstring
        cls_or_func.__doc__ = original_doc + self.docref
        
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            # Directly return the result of the original function or class call
            return cls_or_func(*args, **kwargs)
        
        return wrapper
    
class Deprecated:
    """
    A decorator for marking functions, methods, and classes as deprecated. 
    It emits a deprecation warning when the decorated item is called or 
    instantiated.

    Parameters
    ----------
    reason : str
        The reason why the function, method, or class is deprecated.

    Examples
    --------
    >>> from gofast.decorators import Deprecated
    >>> @Deprecated(reason="Use `new_function` instead.")
    ... def old_function():
    ...     print("This function is deprecated.")
    ...
    >>> old_function()
    # Outputs a deprecation warning and prints: "This function is deprecated."

    Note
    ----
    The warning will point to the location where the deprecated item is 
    used, making it easier to identify and replace deprecated usage in 
    codebases.
    """
    
    def __init__(self, reason):
        if not reason:
            raise ValueError("A reason for deprecation must be supplied.")
        self.reason = reason

    def __call__(self, cls_or_func):
        if not inspect.isfunction(cls_or_func) and not inspect.isclass(cls_or_func):
            raise TypeError("Deprecated decorator can only be applied to functions or classes.")

        fmt = "Call to deprecated {item} {name} ({reason})."
        item_type = "class" if inspect.isclass(cls_or_func) else "function or method"
        msg = fmt.format(item=item_type, name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)

        return new_func

class CheckGDALData:
    """
    A decorator to ensure the availability of GDAL data for functions requiring GDAL. 
    It checks if the GDAL_DATA environment variable is correctly set and points to an 
    existing path. Optionally, it can raise an ImportError if the GDAL data is not 
    configured correctly.

    Parameters
    ----------
    raise_error : bool, optional
        If True, raises an ImportError when GDAL data is not found. Defaults to False.
    verbose : int, optional
        Verbosity level. A higher number indicates more verbose output. Defaults to 0.

    Examples
    --------
    >>> from gofast.decorators import CheckGDALData
    >>> @CheckGDALData(raise_error=True, verbose=1)
    ... def my_gdal_function():
    ...     print("This function uses GDAL.")
    ...
    >>> my_gdal_function()
    # This will either print "This function uses GDAL." if GDAL data is correctly set,
    # or raise an ImportError with instructions on how to configure GDAL data.

    Notes
    -----
    This decorator is particularly useful in environments where GDAL is required but 
    might not be fully configured, such as in some virtual environments or custom 
    installations.
    """

    _has_checked = False
    _gdal_data_found = False

    def __init__(self, raise_error=False, verbose=0):
        self.raise_error = raise_error
        self.verbose = verbose

    def __call__(self, func):
        if not self._has_checked:
            self._check_gdal_data()
            self._has_checked = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._gdal_data_found and self.raise_error:
                raise ImportError(
                    "GDAL is NOT installed correctly. "
                    f"GDAL wheel can be downloaded from {self._gdal_wheel_resources}. "
                    f"See the installation guide: {self._gdal_installation_guide}."
                )
            return func(*args, **kwargs)

        return wrapper

    def _check_gdal_data(self):
        from subprocess import Popen, PIPE 
        if 'GDAL_DATA' in os.environ and os.path.exists(os.environ['GDAL_DATA']):
            if self.verbose:
                _logger.info(f"GDAL_DATA is set to: {os.environ['GDAL_DATA']}")
            self._gdal_data_found = True
        else:
            if self.verbose:
                _logger.warning(
                    "GDAL_DATA environment variable is not set. "
                    f"Please see {self._gdal_data_variable_resources}"
                )
            # Attempt to locate GDAL data using gdal-config
            try:
                if self.verbose:
                    _logger.info("Trying to find gdal-data path ...")
                process = Popen(['gdal-config', '--datadir'], stdout=PIPE, stderr=PIPE)
                output, err = process.communicate()
                if process.returncode == 0 and os.path.exists(output.strip()):
                    os.environ['GDAL_DATA'] = output.strip().decode()
                    if self.verbose:
                        _logger.info(f"Found gdal-data path: {os.environ['GDAL_DATA']}")
                    self._gdal_data_found = True
            except Exception as e:
                if self.verbose:
                    _logger.error(f"Failed to find gdal-data path. Error: {e}")
                self._gdal_data_found = False

    # Class variable declarations for resources and installation guide
    _gdal_data_variable_resources = 'https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable'
    _gdal_wheel_resources = 'https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal'
    _gdal_installation_guide = 'https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/'

class RedirectToNew:
    """
    A decorator to redirect calls from deprecated functions or classes to new ones,
    issuing a deprecation warning and guiding users towards the updated implementation.

    This decorator simplifies the process of transitioning codebases to use new
    functions or classes without breaking existing implementations that rely on
    deprecated ones.

    Parameters
    ----------
    new_target : callable or class
        The new function or class to which calls should be redirected.
    reason : str
        Explanation why the redirection is occurring, typically including deprecation
        information and guidance on using the new target.

    Examples
    --------
    >>> from gofast.decorators import RedirectToNew
    >>> @RedirectToNew(new_function, "Use `new_function` instead of `old_function`.")
    ... def old_function():
    ...     pass
    ...
    >>> old_function()
    # This call will be redirected to `new_function`, with a warning issued about the deprecation.

    """

    def __init__(self, new_target, reason):
        if not callable(new_target):
            raise TypeError("The new target must be a callable or a class.")
        if not isinstance(reason, str):
            raise TypeError("Redirect reason must be supplied as a string.")

        self.new_target = new_target
        self.reason = reason

    def __call__(self, cls_or_func):
        @functools.wraps(cls_or_func)
        def wrapper(*args, **kwargs):
            _logger.warning(f"DEPRECATION WARNING: {self.reason}")
            return self.new_target(*args, **kwargs)

        return wrapper
  
class PlotPrediction:
    """
    A decorator for plotting predictions and observations using matplotlib. 
    This decorator enhances functions that return prediction and observation 
    data by optionally generating a scatter plot for visual comparison.

    Parameters
    ----------
    turn : str, optional
        Controls whether plotting is enabled ('on') or disabled ('off'). Defaults to 'off'.
    **kwargs : dict
        Customization options for the plot, supporting matplotlib.pyplot keywords.

    Attributes
    ----------
    fig_size : tuple
        Figure size for the plot.
    y_pred_kws : dict
        Styling options for the predicted values scatter plot.
    y_obs_kws : dict
        Styling options for the observed values scatter plot.
    tick_params : dict
        Parameters for configuring axis ticks.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    obs_line : tuple
        Controls the observation line plotting ('on', 'off') and its type ('Obs', 'Pred').
    l_kws : dict
        Line properties for the observation line.
    savefig : str or dict
        Path or options for saving the figure.

    Examples
    --------
    >>> from gofast.decorators import PlotPrediction
    >>> @PlotPrediction(turn='on', fig_size=(10, 6))
    ... def my_prediction_function():
    ...     # prediction function logic here
    ...     return y_true, y_pred, 'on'
    ...
    >>> my_prediction_function()
    # This will generate a scatter plot for the predicted and observed values.

    """

    def __init__(self, turn='off', **kwargs):
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (16, 8))
        self.y_pred_kws = kwargs.pop('y_pred_kws', {'c': 'r', 's': 200, 'alpha': 1,
                                                     'label': 'Predicted flow:y_pred'})
        self.y_obs_kws = kwargs.pop('y_obs_kws', {'c': 'blue', 's': 100, 'alpha': 0.8,
                                                   'label': 'Observed flow:y_true'})
        self.tick_params = kwargs.pop('tick_params', {'axis': 'x', 'labelsize': 10,
                                                      'rotation': 90})
        self.xlabel = kwargs.pop('xlabel', 'Boreholes tested')
        self.ylabel = kwargs.pop('ylabel', 'Flow rates(FR) classes')
        self.obs_line = kwargs.pop('obs_line', ('off', 'Obs'))
        self.l_kws = kwargs.pop('l_kws', {'c': 'blue', 'ls': '--', 'lw': 1, 'alpha': 0.5})
        self.savefig = kwargs.pop('savefig', None)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            y_true, y_pred, switch = func(*args, **kwargs)
            turn = switch if switch is not None else self.turn

            if turn == 'on':
                self._plot(y_true, y_pred)

            return y_true, y_pred, switch

        return wrapper

    def _plot(self, y_true, y_pred):
        plt.figure(figsize=self.fig_size)
        plt.scatter(y_true.index, y_true, **self.y_obs_kws)
        plt.scatter(y_pred.index, y_pred, **self.y_pred_kws)

        if self.obs_line[0] == 'on':
            data = y_true if 'true' in self.obs_line[1].lower(
                ) or 'obs' in self.obs_line[1].lower() else y_pred
            plt.plot(data.index, data, **self.l_kws)

        plt.tick_params(**self.tick_params)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()

        if self.savefig:
            if isinstance(self.savefig, str):
                plt.savefig(self.savefig)
            else:
                plt.savefig(**self.savefig)

        plt.show()

class PlotFeatureImportance:
    """
    A decorator to plot permutation feature importance (PFI) diagrams or dendrogram 
    figures for feature correlation analysis. It utilizes matplotlib for plotting 
    and can be customized with various keyword arguments.

    Parameters
    ----------
    kind : str, optional
        Specifies the type of plot to generate. Options are:
        - 'pfi' for permutation feature importance before and after shuffling trees.
        - 'dendro' for a dendrogram plot showing feature correlations.
        Defaults to 'pfi'.
    turn : str, optional
        Controls whether to plot ('on') or not ('off'). Defaults to 'off'.
    **kwargs : dict
        Keyword arguments for matplotlib plotting functions and additional customization.

    Examples
    --------
    >>> from gofast.decorators import PlotFeatureImportance
    >>> @PlotFeatureImportance(kind='pfi', turn='on', fig_size=(10, 6))
    ... def my_model_analysis_function():
    ...     # Function logic here
    ...     return X, y_pred, y_true, model, feature_names, 'on'
    ...
    >>> my_model_analysis_function()
    # This will plot the specified PFI diagram if turn is 'on'.

    Note
    ----
    Ensure matplotlib is installed in your environment to use this decorator.
    """
    
    def __init__(self, kind='pfi', turn='off', **kwargs):
        self.kind = kind
        self.turn = turn
        self.fig_size = kwargs.pop('fig_size', (9, 3))
        self.savefig = kwargs.pop('savefig', None)
        # Default keyword arguments for various plots
        self.barh_kws = kwargs.pop('barh_kws', {'color': 'blue', 'edgecolor': 'k', 'linewidth': 2})
        self.box_kws = kwargs.pop('box_kws', {'vert': False, 'patch_artist': True})
        self.dendro_kws = kwargs.pop('dendro_kws', {'leaf_rotation': 90})
        self.plot_kwargs = kwargs  # Remaining kwargs for further customization

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the decorated function
            results = func(*args, **kwargs)
            # Unpack results based on expected structure
            X, y_pred, y_true, model, feature_names, switch = results
            
            # Update settings based on function return values if provided
            self.turn = switch if switch is not None else self.turn
            
            # Proceed to plot if enabled
            if self.turn.lower() == 'on':
                self._plot_results(X, y_pred, y_true, model, feature_names)
            
            return results

        return wrapper

    def _plot_results(self, X, y_pred, y_true, model, feature_names):
        if self.kind == 'pfi':
            self._plot_pfi(X, model, feature_names)
        elif self.kind == 'dendro':
            self._plot_dendrogram(X, feature_names)
        else:
            warnings.warn(f"Unknown kind '{self.kind}'. No plot will be generated.")

        if self.savefig:
            plt.savefig(self.savefig, **self.plot_kwargs)

    def _plot_pfi(self, X, model, feature_names):
        # Example PFI plotting.
        plt.figure(figsize=self.fig_size)
        plt.barh(range(len(feature_names)), model.feature_importances_, **self.barh_kws)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.show()

    def _plot_dendrogram(self, X, feature_names):
        # Example dendrogram plotting.
        from scipy.cluster.hierarchy import dendrogram, linkage
        Z = linkage(X, 'ward')
        plt.figure(figsize=self.fig_size)
        dendrogram(Z, labels=feature_names, **self.dendro_kws)
        plt.show()

class AppendDocSection:
    """
    A decorator to append a specific section of a function's or class's docstring
    to another. This is particularly useful for avoiding redundancy when documenting
    shared parameters or information across multiple functions or classes.

    Parameters
    ----------
    source_func : callable
        The source function or class whose docstring part will be appended.
    start : str, optional
        The start marker from where to begin appending the docstring.
    end : str, optional
        The end marker where to stop appending the docstring. If not provided, 
        everything from the `start` to the end of the source's docstring is appended.

    Examples
    --------
    >>> from gofast.decorators import AppendDocSection
    >>> @AppendDocSection(source_func=writedf, start='param reason', end='param to_')
    ... def new_function():
    ...     '''Function-specific docstring.'''
    ...     pass
    ...
    >>> print(new_function.__doc__)
    # This will include the section of `writedf`'s docstring from 'param reason' 
    # to 'param to_' appended to 'new_function' docstring.

    """
    
    def __init__(self, source_func, start=None, end=None):
        if not callable(source_func):
            raise TypeError("`source_func` must be a callable.")
        self.source_func = source_func
        self.start = start
        self.end = end

    def __call__(self, target_func):
        source_doc = inspect.getdoc(self.source_func) or ''
        target_doc = inspect.getdoc(target_func) or ''
        
        # Find the start index
        start_ix = source_doc.find(self.start) if self.start else 0
        end_ix = source_doc.find(self.end, start_ix) if self.end else len(source_doc)
        
        # Handle cases where start or end markers are not found
        if self.start and start_ix == -1:
            warnings.warn(f"Start marker '{self.start}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            start_ix = 0
        if self.end and end_ix == -1:
            warnings.warn(f"End marker '{self.end}' not found in"
                          f" `{self.source_func.__name__}` docstring.")
            end_ix = len(source_doc)

        # Extract the desired docstring section
        doc_section = source_doc[start_ix:end_ix]

        # Append the extracted section to the target's docstring
        target_func.__doc__ = (target_doc + "\n\n" + doc_section).strip()

        return target_func

class AppendDocFrom:
    """
    A decorator for appending a specific section of a function's or class's docstring
    to another. This is useful for avoiding redundancy in documentation, especially
    for shared parameters or descriptions.

    Parameters
    ----------
    source : callable
        The source function or class from which to extract the docstring section.
    from_ : str
        The start marker for the docstring section to be extracted.
    to : str, optional
        The end marker for the docstring section. If not provided, everything
        from `from_` to the end of the source's docstring is used.
    insert_at : str
        The marker in the target's docstring where the extracted section should
        be inserted. If not found, the section is appended at the end.

    Examples
    --------
    >>> from gofast.decorators import AppendDocFrom
    >>> @AppendDocFrom(source=func0, from_='Parameters', to='Returns', insert_at='Parameters')
    ... def new_func():
    ...     '''New function docstring.'''
    ...     pass
    ...
    >>> print(new_func.__doc__)
    # This will print 'new_func' docstring with the 'func0' docstring section 
    # from 'Parameters' to 'Returns' appended at the 'Parameters' marker.

    Note
    ----
    It's recommended to use docstrings with consistent formatting to ensure
    proper insertion and readability.

    """
    def __init__(self, source, from_, to=None, insert_at='Parameters'):
        self.source = source
        self.from_ = from_
        self.to = to
        self.insert_at = insert_at.lower()
        
    def __call__(self, target):
        self._append_doc(target)
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            return target(*args, **kwargs)

        # self._append_doc(target)
        return wrapper
    
    def _append_doc(self, target):
        source_doc = inspect.getdoc(self.source) or ''
        target_doc = inspect.getdoc(target) or ''
        
        start_index = source_doc.find(self.from_)
        end_index = source_doc.find(self.to, start_index) if self.to else len(source_doc)
        
        if start_index == -1 or (self.to and end_index == -1):
            warnings.warn(f"Cannot find specified docstring section in `{self.source.__name__}`.")
            return
        
        doc_section = source_doc[start_index:end_index]
        
        insert_index = target_doc.lower().find(self.insert_at)
        if insert_index == -1:
            target_doc += "\n\n" + doc_section
        else:
            part1 = target_doc[:insert_index]
            part2 = target_doc[insert_index:]
            target_doc = part1 + doc_section + "\n\n" + part2
        
        target.__doc__ = target_doc

class NumpyDocstring:
    """
    A class decorator designed to automatically parse and reformat the docstring of
    the decorated function into a structured NumPy-style docstring. This decorator
    enhances readability and standardization of documentation, making it more useful
    for developers and users, especially in the context of Sphinx-generated docs.

    Parameters
    ----------
    func : function, optional
        The function to be decorated. If not provided at initialization, it must be
        set later by calling the instance as a decorator.
    enforce_strict : bool, optional
        If True, enforces strict NumPy docstring formatting rules. This may include
        checking for specific sections and their order. Defaults to False, allowing
        for more flexibility in the original docstring format.
    custom_sections : dict, optional
        A dictionary where keys are section titles (e.g., "Custom Section") and values
        are the content for those sections. This allows for the addition of custom
        sections not typically found in NumPy docstrings.

    Examples
    --------
    Using as a decorator directly on a function:

    @NumpyDocstring
    def my_function(x, y):
        \"\"\"Function docstring.\"\"\"
        return x + y

    Adding custom sections and enforcing strict formatting:

    @NumpyDocstring(enforce_strict=True, custom_sections={
        'Custom Section': 'Details here.'})
    def another_function(x):
        \"\"\"Another function docstring.\"\"\"
        return x * 2
    """

    def __init__(self, func=None, *, enforce_strict=False, custom_sections=None):
        self.func = func
        self.enforce_strict = enforce_strict
        self.custom_sections = custom_sections or {}
        if func is not None:
            functools.update_wrapper(self, func)
            self._update_docstring()

    def __call__(self, *args, **kwargs):
        if self.func:
            return self.func(*args, **kwargs)
        else:
            def wrapper(func):
                self.func = func
                functools.update_wrapper(self, func)
                self._update_docstring()
                return self
            return wrapper

    def __get__(self, instance, owner):
        return self if instance is None else functools.partial(self.__call__, instance)

    def _parse_docstring(self, docstring):
        """
        Advanced parsing of the original docstring to identify and reformat sections.
        """
        # Define the sections and their possible headings in docstrings
        section_headings = {
            'Parameters': ['parameters', 'args', 'arguments', ':param'],
            'Returns': ['returns', ':return', ':returns'],
            'Raises': ['raises', ':raise', ':raises'],
            'Examples': ['examples', ':example'],
            'Warnings': ['warnings', ':warning'],
            'See Also': ['see also', 'references'],
            'Notes': ['notes']
        }

        # Initialize sections dictionary
        sections = {key: '' for key in section_headings}

        # Regular expression to detect section headings
        section_regex = re.compile(
            r'^\s*(?P<section>' + '|'.join(
                [f"(?:{'|'.join(headings)})" for headings in section_headings.values()]
                ) + r')\s*$', re.IGNORECASE)

        current_section = None
        for line in docstring.split('\n'):
            match = section_regex.match(line.strip())
            if match:
                # Find which section it belongs to
                for section, headings in section_headings.items():
                    if match.group('section').lower() in headings:
                        current_section = section
                        break
            elif current_section:
                sections[current_section] += line + '\n'

        # Apply custom sections if any
        for section, content in self.custom_sections.items():
            sections[section] = content

        return sections

    def _format_section(self, title, content):
        """
        Format a single section with the given title and content.
        """
        if not content.strip():
            return ''
        return f"{title}\n{'-' * len(title)}\n{content.strip()}\n"

    def _update_docstring(self):
        """
        Update the function's docstring with parsed and formatted content.
        """
        if not self.func.__doc__:
            return

        sections = self._parse_docstring(self.func.__doc__)
        formatted_docstring = "\n".join(self._format_section(
            title, content) for title, content in sections.items() if content)

        self.func.__doc__ = formatted_docstring

    def __set_name__(self, owner, name):
        self._update_docstring()
        setattr(owner, name, self)
        
def sanitize_docstring(enforce_strict=False, custom_sections=None):
    """
    Decorator factory function that returns an instance of NumpyDocstring.
    This function simplifies the application of the decorator with additional parameters
    like enforcing strict formatting and adding custom sections to the docstring.

    Parameters
    ----------
    enforce_strict : bool, optional
        If set to True, the decorator enforces strict adherence to the NumPy docstring
        format, potentially raising errors for non-compliance. Defaults to False.
    custom_sections : dict, optional
        Allows for the specification of custom sections in the decorated function's
        docstring. Keys are the titles of the sections, and values are the content.

    Returns
    -------
    decorator : NumpyDocstring
        An instance of AdvancedNumpyDocDecorator configured with the provided parameters.

    Examples
    --------
    Decorating a function with custom sections and without strict enforcement:

    @sanitize_docstring(custom_sections={'Custom Usage': 'This is how you use this function.'})
    def sample_function(param1, param2):
        \"\"\"This function does something interesting.\"\"\"
        pass
    
    """
    def decorator(func):
        return NumpyDocstring(func, enforce_strict=enforce_strict,
                              custom_sections=custom_sections)
    return decorator

class SuppressOutput:
    """
    A context manager for suppressing stdout and stderr messages. It can be
    useful when interacting with APIs or third-party libraries that output
    messages to the console, and you want to prevent those messages from
    cluttering your output.

    Parameters
    ----------
    suppress_stdout : bool, optional
        Whether to suppress stdout messages. Default is True.
    suppress_stderr : bool, optional
        Whether to suppress stderr messages. Default is True.

    Examples
    --------
    >>> from gofast.decorators import SuppressOutput
    >>> with SuppressOutput():
    ...     print("This will not be printed to stdout.")
    ...     raise ValueError("This error message will not be printed to stderr.")
    
    Note
    ----
    This class is particularly useful in scenarios where controlling external
    library output is necessary to maintain clean and readable application logs.

    See Also
    --------
    contextlib.redirect_stdout, contextlib.redirect_stderr : For more granular control
    over output redirection in specific parts of your code.
    """
    
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
        self._devnull = None

    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = self._devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self._stdout is not None:
            sys.stdout = self._stdout
        if self.suppress_stderr and self._stderr is not None:
            sys.stderr = self._stderr
        if self._devnull is not None:
            self._devnull.close()

class _M:
    def _m(self): pass
MethodType = type(_M()._m)

class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        functools.update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            @functools.wraps(self.fn)
            def out(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

        return out

def available_if(check):
    """An attribute that is available only if check returns a truthy value

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Examples
    --------
    >>> from sklearn.tools.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)


def dataify(func: Callable) -> Callable:
    """
    A decorator that ensures the first positional argument passed to the 
    decorated function is a pandas DataFrame.
    
    If the argument is not a DataFrame, the decorator attempts to convert it 
    into one using an optional 'columns' keyword argument.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function with data conversion logic.

    Notes
    -----
    The decorated function must accept its first positional argument as data
    and may optionally accept a 'columns' keyword argument to specify column names
    for the DataFrame conversion.

    Examples
    --------
    >>> from gofast.decorators import dataify
    >>> @dataify
    ... def my_function(data, /, columns=None, **kwargs):
    ...     print(data)
    ...     print("Columns:", columns)
    >>> import numpy as np
    >>> my_function(np.array([[1, 2], [3, 4]]), columns=['A', 'B'])
       A  B
    0  1  2
    1  3  4
    Columns: ['A', 'B']
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data = args[0]
        columns = kwargs.get('columns', None)
        # Check if the first positional argument is not a DataFrame
        if not isinstance(data, pd.DataFrame):
            # Attempt to convert it into a DataFrame
            try:
                data = pd.DataFrame(data, columns=columns)
                # If columns are provided but do not match data dimensions, ignore them
                if columns and len(columns) != data.shape[1]:
                    data = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Error converting data to DataFrame: {e}")

            # Call the decorated function with the new DataFrame 
            # as the first argument
            return func(data, *args[1:], **kwargs)
        else:
            # If the first argument is already a DataFrame, 
            # proceed as normal
            return func(*args, **kwargs)
    return wrapper

class NumpyDocstringFormatter:
    """
    A decorator class for reformatting function docstrings to adhere to the
    NumPy documentation standard.

    This class provides a flexible way to ensure that the docstrings of 
    decorated functions follow a consistent format, making them more readable
    and compatible with tools like Sphinx for generating documentation. It can
    automatically extract and reformat specified sections of a docstring, and
    optionally validate the result using Sphinx.

    Parameters
    ----------
    include_sections : list of str, optional
        A list of section names to include in the reformatted docstring. 
        If None (the default), all recognized sections are included. This 
        allows for selective inclusion of sections like "Parameters", "Returns",
        "Examples", etc., based on user preference or requirements.
        
        Example: ['Parameters', 'Returns', 'Examples']

    validate_with_sphinx : bool, default False
        Indicates whether the reformatted docstring should be validated using
        Sphinx. This can be useful for ensuring that the docstring is not only
        correctly formatted but also compatible with Sphinx documentation
        generation. Note that actual implementation of Sphinx validation is 
        not provided in this example and would require integration with Sphinx's
        documentation building process.

    custom_formatting : callable, optional
        A custom function that applies additional formatting to each section
        of the docstring. This function should accept two arguments: 
            `section_name` (a string indicating the name of the section) and 
            `section_content` (the content of the section as a string), and 
            return the formatted content as a string. This allows for
        further customization of the docstring formatting beyond the standard
        reformatting performed by this class::
        
        Example function:
            def custom_formatter(section_name, section_content):
                # Custom formatting logic here
                return formatted_content

    Examples
    --------
    Using the decorator with default settings to reformat all sections:

    >>> from gofast.decorators import NumpyDocstringFormatter
    >>> @NumpyDocstringFormatter()
    ... def example_function(param1, param2=None):
    ...     '''
    ...     This is an example function with parameters.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : int
    ...         The first parameter.
    ...     param2 : int, optional
    ...         The second parameter (default is None).
    ...     '''
    ...     return True

    Specifying sections to include and enabling Sphinx validation:

    >>> @NumpyDocstringFormatter(include_sections=['Parameters', 'Returns'], 
    ...                             validate_with_sphinx=True)
    ... def another_function(param1):
    ...     '''
    ...     Another example function demonstrating selective section inclusion
    ...     and Sphinx validation.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter.
    ...
    ...     Returns
    ...     -------
    ...     bool
    ...         Always returns True.
    ...     '''
    ...     return True

    Applying custom formatting to docstring sections:

    >>> def uppercase_formatter(section_name, section_content):
    ...     # Example custom formatting function that uppercases section content
    ...     return section_content.upper()
    ...
    >>> @NumpyDocstringFormatter(custom_formatting=uppercase_formatter)
    ... def custom_formatted_function(param1):
    ...     '''
    ...     Function demonstrating custom formatting of docstring sections.
    ...
    ...     Parameters
    ...     ----------
    ...     param1 : str
    ...         A string parameter to be uppercased in the documentation.
    ...     '''
    ...     return param1.upper()
    """

    def __init__(self, include_sections=None, validate_with_sphinx=False, 
                 custom_formatting=None, verbose=0):

        self.include_sections = include_sections
        self.validate_with_sphinx = validate_with_sphinx
        self.custom_formatting = custom_formatting
        self.verbose=verbose 

    def __call__(self, func):
        """
        Decorator method to apply the docstring formatting.

        Parameters
        ----------
        func : function
            The function to decorate.

        Returns
        -------
        function
            The decorated function with a reformatted docstring.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = self.format_docstring(func.__doc__)
        if self.validate_with_sphinx:
            self.sphinx_validation(wrapper.__doc__)
        
        return wrapper

    def format_docstring(self, docstring):
        """
        

        Parameters
        ----------
        docstring : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        """
        Format the original docstring according to the NumPy standard.

        Parameters
        ----------
        docstring : str
            The original docstring to format.

        Returns
        -------
        str
            The formatted docstring.
        """
        if not docstring:
            return ''

        sections_order = ['Parameters', 'Returns', 'Raises', 'Examples', 
                          'Warnings', 'See Also', 'Notes']
        if self.include_sections is not None:
            sections_order = [s for s in sections_order if s in self.include_sections]

        sections = self.extract_sections(docstring, sections_order)
        formatted_docstring = self.reconstruct_docstring(sections, sections_order)

        return formatted_docstring

    def extract_sections(self, docstring, sections_order):
        """
        Extract and return the docstring sections found in the given order.

        Parameters
        ----------
        docstring : str
            The original docstring from which to extract sections.
        sections_order : list of str
            The ordered list of section names to extract.

        Returns
        -------
        dict
            A dictionary with section names as keys and extracted content as values.
        """
        sections = {section: '' for section in sections_order}
        # Simplified patterns
        section_patterns = {
            'Parameters': re.compile(r'parameters\s*[\n\r]+', re.IGNORECASE),
            'Returns': re.compile(r'returns\s*[\n\r]+', re.IGNORECASE),
            'Raises': re.compile(r'raise[s]?\s*[\n\r]+', re.IGNORECASE),
            'Examples': re.compile(r':examples:\s*[\n\r]+', re.IGNORECASE),
            'Warnings': re.compile(r'warnings\s*[\n\r]+', re.IGNORECASE),
            'See Also': re.compile(r'see also\s*[\n\r]+', re.IGNORECASE),
            'Notes': re.compile(r'notes\s*[\n\r]+', re.IGNORECASE),
        }

        for section in sections_order:
            if section in section_patterns:
                match = section_patterns[section].search(docstring)
                if match:
                    content = match.group(0).strip()
                    if self.custom_formatting:
                        content = self.custom_formatting(section, content)
                    sections[section] = content

        return sections

    def reconstruct_docstring(self, sections, sections_order):
        """
        Reconstruct the docstring from the extracted sections in the given order.

        Parameters
        ----------
        sections : dict
            The sections extracted from the original docstring.
        sections_order : list of str
            The ordered list of section names to include in the reconstructed docstring.

        Returns
        -------
        str
            The reconstructed docstring.
        """
        reconstructed_docstring = ""
        for section in sections_order:
            if sections[section]:
                reconstructed_docstring += f"{section}\n{'-' * len(section)}\n{sections[section]}\n\n"
        return reconstructed_docstring
    
    def sphinx_validation(self, docstring):
        """
        Validates the given docstring using Sphinx and docutils to ensure 
        it adheres to standards acceptable by Sphinx for documentation generation.
    
        Parameters
        ----------
        docstring : str
            The docstring to validate.
    
        Note
        ----
        This method provides a conceptual approach and requires a Sphinx 
        environment to be properly implemented.
        """
        from docutils import nodes
        from docutils.core import publish_doctree
        from .tools._dependency import import_optional_dependency
        
        try: 
            import_optional_dependency ("docutils")
        except: 
            from .tools.funcutils import install_package
            install_package('docutils' )
        try:
            # Create a new document for parsing
            settings_overrides = {'report_level': 2, 'warning_stream': False}
            document = publish_doctree(docstring, settings_overrides=settings_overrides)
            
            # Check for any errors or warnings in the parsed document
            warnings_or_errors = document.traverse(condition=lambda node: isinstance(
                node, (nodes.warning, nodes.error)))
            if next(warnings_or_errors, None):
                if self.verbose:
                    _logger.warning(
                        "Docstring validation failed with warnings or errors.") 
            else:
                if self.verbose:
                    _logger.info("Docstring passed Sphinx validation.")
        except Exception as e:
            if self.verbose:
                _logger.error(
                    f"Docstring validation failed due to an exception: {e}") 

class Dataify:
    """
    A class decorator that ensures the first positional argument passed 
    to the decorated function is a pandas DataFrame, offering flexibility 
    through additional parameters for various data handling scenarios.

    Parameters
    ----------
    enforce_dataframe : bool, optional
        Whether to enforce the conversion of the first positional argument 
        to a pandas DataFrame. Defaults to True.
    columns : list of str, optional
        Specifies the column names for DataFrame conversion. If not provided, 
        and data conversion is necessary, default integer column names are used. 
        This parameter is considered only if `enforce_dataframe` is True.
    ignore_mismatch : bool, optional
        If True, ignores the `columns` parameter if its length does not match 
        the data dimensions, using default integer column names instead. 
        Defaults to False.
    fail_silently : bool, optional
        If True, the decorator will not raise an exception if the conversion 
        fails, and will instead pass the original data to the function. 
        Defaults to False.

    Examples
    --------
    >>> from gofast.decorators import Dataify
    >>> @Dataify(enforce_dataframe=True, columns=['A', 'B'], ignore_mismatch=True)
    ... def process_data(data):
    ...     print(data)

    >>> import numpy as np
    >>> process_data(np.array([[1, 2], [3, 4]]))
       A  B
    0  1  2
    1  3  4

    Notes
    -----
    - The decorated function must accept its first positional argument as data.
    - This class is beneficial for functions expected to work with data in 
      pandas DataFrame format, automating input data conformity checks.
    """

    def __init__(
        self, enforce_dataframe=True, 
        columns=None, 
        ignore_mismatch=False, 
        fail_silently=False
        ):
        self.enforce_dataframe = enforce_dataframe
        self.columns = columns
        self.ignore_mismatch = ignore_mismatch
        self.fail_silently = fail_silently

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enforce_dataframe or not args:
                return func(*args, **kwargs)

            data = args[0]
            if not isinstance(data, pd.DataFrame):
                try:
                    data = self._attempt_dataframe_conversion(data, **kwargs)
                    
                except ValueError as e:
                    if self.fail_silently:
                        warnings.warn(f"Dataify Warning: {e}")
                        return func(*args, **kwargs)
                    else:
                        raise

            return func(data, *args[1:], **kwargs)
        return wrapper

    def _attempt_dataframe_conversion(self, data, **kwargs):
        """
        Attempts to convert the input data to a pandas DataFrame using 
        the specified columns if applicable, handling dimension mismatches.

        Parameters
        ----------
        data : array-like, Iterable, dict, or DataFrame
            The data to convert to a DataFrame.
        **kwargs : dict
            Additional keyword arguments passed to the decorated function, 
            potentially including 'columns' for specifying DataFrame column 
            names.

        Returns
        -------
        pd.DataFrame
            The data converted to a pandas DataFrame.

        Raises
        ------
        ValueError
            If the conversion fails due to incompatible data or column 
            specifications, unless `fail_silently` is True.

        Notes
        -----
        This method is a private helper intended for internal use by the 
        Dataify decorator to manage DataFrame conversion.
        """
        columns = kwargs.get('columns', self.columns)
        try:
            if columns is not None and not self.ignore_mismatch:
                return pd.DataFrame(data, columns=columns )
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error converting data to DataFrame: {e}")

# Example usage
@NumpyDocstringFormatter(include_sections=['Parameters', 'Returns'], validate_with_sphinx=True)
def example_function(param1, param2=None):
    """
    This is an example function that demonstrates the usage of the NumpyDocstringFormatter.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : int, optional
        The second parameter (default is None).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    return True
    
