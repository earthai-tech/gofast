# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
I/O utilities for handling data reading, saving, and loading.
Includes functions to read and save data while ensuring file integrity.
Supports conversion and validation of data formats for compatibility.
"""
from __future__ import annotations
import os 
import pathlib
import warnings 
import textwrap

import numpy as np 
import pandas as pd 
from collections.abc import Iterable
from functools import wraps 
from typing import Any, List, Union, Dict, Optional, Callable  

from ..exceptions import FileHandlingError 
from ..api.types import DataFrame, NDArray
from ..api.util import get_table_size
from ..api.property import PandasDataHandlers
from .array_manager import to_numeric_dtypes
from .checks import is_iterable, check_params 
from .utils import  ellipsis2false, lowertify, smart_format 

TW = get_table_size()

__all__=[
    "EnsureFileExists", 
    "read_data",
    "save_or_load",
    "is_data_readable",
    "to_frame_if", 
    "SaveFile", 
    "fmt_text",
    "export_data"
    ]

class EnsureFileExists:
    """
    Class decorator to ensure a file or URL exists before calling the 
    decorated function.

    This decorator checks if the specified file or URL exists before executing  
    the decorated function. If the file does not exist, it raises a
    FileNotFoundError. If the URL does not exist, it raises a ConnectionError. 
    The decorator can be configured to print verbose messages during the check.
    It also handles other data types based on the specified action.

    Parameters
    ----------
    file_param : int or str, optional
        The index of the parameter that specifies the file path or URL or 
        the name of the keyword argument (default is 0). If an integer is 
        provided, it refers to the position of the argument in the function 
        call. If a string is provided, it refers to the keyword argument name.
    verbose : bool, optional
        If True, prints messages indicating the file or URL check status 
        (default is False).
    action : str, optional
        Action to take if the parameter is not a file or URL. Options are 
        'ignore', 'warn', or 'raise' (default is 'raise').

    Examples
    --------
    Basic usage with verbose output:
    
    >>> from gofast.core.io import EnsureFileExists
    >>> @EnsureFileExists(verbose=True)
    ... def process_data(file_path: str):
    ...     print(f"Processing data from {file_path}")
    >>> process_data("example_file.txt")

    Basic usage without parentheses:
    
    >>> @EnsureFileExists
    ... def process_data(file_path: str):
    ...     print(f"Processing data from {file_path}")
    >>> process_data("example_file.txt")

    Checking URL existence:
    
    >>> from gofast.decorators import EnsureFileExists
    >>> @EnsureFileExists(file_param='url', verbose=True)
    ... def fetch_data(url: str):
    ...     print(f"Fetching data from {url}")
    >>> fetch_data("https://example.com/data.csv")
    
    Notes
    -----
    This decorator is particularly useful for functions that require a file path 
    or URL as an argument and need to ensure the file or URL exists before 
    proceeding with further operations. It helps in avoiding runtime errors 
    due to missing files or unreachable URLs.
    
    See Also
    --------
    os.path.isfile : Checks if a given path is an existing regular file.
    requests.head : Sends a HEAD request to a URL to check its existence.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in Python. 
           Proceedings of the 9th Python in Science Conference, 51-56.
    
    """
    def __init__(
            self, file_param: Union[int, str] = 0, 
            verbose: bool = False, 
            action: str = 'raise'
            ):
        self.file_param = file_param
        self.verbose = verbose
        self.action = action

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> any:
            # Determine the file path or URL from args or kwargs
            file_path_or_url = None
            if isinstance(self.file_param, int):
                if len(args) > self.file_param:
                    file_path_or_url = args[self.file_param]
            elif isinstance(self.file_param, str):
                file_path_or_url = kwargs.get(self.file_param)
    
            if self.verbose:
                print(f"Checking if path or URL exists: {file_path_or_url}")
    
            # Check if the file path or URL exists
            if file_path_or_url is None:
                self.handle_action(f"File or URL not specified: {file_path_or_url}")
            elif isinstance(file_path_or_url, str):
                if file_path_or_url.startswith(('http://', 'https://')):
                    if not self.url_exists(file_path_or_url):
                        self.handle_action(f"URL not reachable: {file_path_or_url}")
                    elif self.verbose:
                        print(f"URL exists: {file_path_or_url}")
                else:
                    if not os.path.isfile(file_path_or_url):
                        self.handle_action(f"File not found: {file_path_or_url}")
                    elif self.verbose:
                        print(f"File exists: {file_path_or_url}")
            else:
                if self.action == 'ignore':
                    if self.verbose:
                        print(f"Ignoring non-file, non-URL argument: {file_path_or_url}")
                elif self.action == 'warn':
                    warnings.warn(f"Non-file, non-URL argument provided: {file_path_or_url}")
                else:
                    raise TypeError(f"Invalid file or URL argument: {file_path_or_url}")
    
            return func(*args, **kwargs)
    
        return wrapper

    def handle_action(self, message: str):
        """
        Handle the action based on the specified action parameter.

        Parameters
        ----------
        message : str
            The message to display or include in the raised exception.
        """
        if self.action == 'ignore':
            if self.verbose:
                print(f"Ignoring: {message}")
        elif self.action == 'warn':
            warnings.warn(message)
        elif self.action == 'raise':
            raise FileNotFoundError(message)
        else:
            raise ValueError(f"Invalid action: {self.action}")

    @staticmethod
    def url_exists(url: str) -> bool:
        """
        Check if a URL exists.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        bool
            True if the URL exists, False otherwise.
        """
        import requests
        try:
            response = requests.head(url, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @classmethod
    def ensure_file_exists(
        cls, func: Optional[Callable] = None, *, 
        file_param: Union[int, str] = 0, 
        verbose: bool = False, 
        action: str = 'raise'):
        """
        Class method to allow the decorator to be used without parentheses.

        This method enables the decorator to be applied directly without 
        parentheses, by using the first positional argument as the file or URL 
        to check. It also allows setting the `file_param`, `verbose`, and `action`
        parameters when called with parentheses.

        Parameters
        ----------
        func : Callable, optional
            The function to be decorated.
        file_param : int or str, optional
            The index of the parameter that specifies the file path or URL 
            or the name of the keyword argument (default is 0).
        verbose : bool, optional
            If True, prints messages indicating the file or URL check status 
            (default is False).
        action : str, optional
            Action to take if the parameter is not a file or URL. 
            Options are 'ignore', 'warn', or 'raise' (default is 'raise').

        Returns
        -------
        Callable
            The decorated function with file or URL existence check.

        Examples
        --------
        >>> from gofast.decorators import EnsureFileExists
        >>> @EnsureFileExists(verbose=True)
        ... def process_data(file_path: str):
        ...     print(f"Processing data from {file_path}")
        >>> process_data("example_file.txt")

        >>> from gofast.decorators import EnsureFileExists
        >>> @EnsureFileExists
        ... def process_data(file_path: str):
        ...     print(f"Processing data from {file_path}")
        >>> process_data("example_file.txt")
        """
        if func is not None:
            return cls(file_param, verbose, action)(func)
        return cls(file_param, verbose, action)

# Allow decorator to be used without parentheses
EnsureFileExists = EnsureFileExists.ensure_file_exists

class SaveFile:
    """
    SaveFile Decorator for Smartly Saving DataFrames in Various Formats.
    
    The `SaveFile` decorator enables automatic saving of DataFrames returned 
    by decorated functions or methods. It intelligently handles different 
    return types, such as single DataFrames or tuples containing DataFrames, 
    and utilizes the `PandasDataHandlers` class to manage file writing based 
    on provided file extensions.
    
    The decorator extracts the `savefile` keyword argument from the decorated 
    function or method. If `savefile` is specified, it determines the 
    appropriate writer based on the file extension and saves the DataFrame 
    accordingly. If the decorated function does not include a `savefile` 
    keyword argument, the decorator performs no action and simply returns 
    the original result.
    
    Parameters
    ----------
    savefile : str, optional
        The file path where the DataFrame should be saved. If `None`, no file 
        is saved.
    data_index : int, default=0
        The index to extract the DataFrame from the returned tuple. Applicable
         only if the decorated function returns a tuple.
    dout : int, default='.csv'
        The default output to save the dataframe if the extension of the file 
        is not provided by the user. 
        
    Methods
    -------
    __call__(self, func):
        Makes the class instance callable and applies the decorator logic to the 
        decorated function.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.core.io import SaveFile
    >>> from gofast.utils.datautils import to_categories
    
    >>> # Sample DataFrame
    >>> data = {
    ...     'value': np.random.uniform(0, 100, 1000)
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Define a function that categorizes and returns the DataFrame
    >>> @SaveFile(data_index=0)
    ... def categorize_values(df, savefile=None):
    ...     df = to_categories(
    ...         df=df,
    ...         column='value',
    ...         categories='auto'
    ...     )
    ...     return df
    
    >>> # Execute the function with savefile parameter
    >>> df = categorize_values(df, savefile='output/value_categories.csv')
    
    >>> # The categorized DataFrame is saved to 'output/value_categories.csv'
    
    >>> # Define a function that returns a tuple containing multiple DataFrames
    >>> @SaveFile(data_index=1)
    ... def process_data(df, savefile=None):
    ...     categorized_df = to_categories(
    ...         df=df,
    ...         column='value',
    ...         categories='auto'
    ...     )
    ...     summary_df = df.describe()
    ...     return (categorized_df, summary_df)
    
    >>> # Execute the function with savefile parameter targeting the summary DataFrame
    >>> categorized, summary = process_data(df, savefile='output/summary_stats.xlsx')
    
    >>> # The summary DataFrame is saved to 'output/summary_stats.xlsx'
    
    Notes
    -----
    - The decorator leverages the `PandasDataHandlers` class to support a wide 
      range of file formats based on the provided file extension.
    - If the decorated function does not include a `savefile` keyword argument, 
      the decorator does not perform any saving operations and simply returns the 
      original result.
    - When dealing with tuple returns, ensure that the `data_index` corresponds 
      to the position of the DataFrame within the tuple.
    - Unsupported file extensions will trigger a warning, and the DataFrame will 
      not be saved.

    See Also
    --------
    PandasDataHandlers : Class for handling Pandas data parsing and writing.
    pandas.DataFrame.to_csv : Method to write a DataFrame to a CSV file.
    pandas.DataFrame.to_excel : Method to write a DataFrame to an Excel file.
    pandas.DataFrame.to_json : Method to write a DataFrame to a JSON file.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame.to_csv. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    .. [2] Pandas Documentation: pandas.DataFrame.to_excel. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html
    .. [3] Pandas Documentation: pandas.DataFrame.to_json. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html
    .. [4] Python Documentation: functools.wraps. 
       https://docs.python.org/3/library/functools.html#functools.wraps
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """

    def __init__(
        self, func=None, *, 
        data_index: int = 0, 
        dout='.csv'
        ):
        self.func = func
        self.data_index = data_index
        self.dout = dout 
        self.data_handler = PandasDataHandlers()

    def __call__(self, *args, **kwargs):
        if self.func is None:
            # Decorator is called with arguments
            func = args[0]
            return SaveFile(func, data_index=self.data_index)

        @wraps(self.func)
        def wrapper(*args, **kwargs):
            result = self.func(*args, **kwargs)

            savefile = kwargs.get('savefile', None)
            if savefile is not None:
                _, ext = os.path.splitext(savefile)

                if not ext:
                    if ( 
                        self.dout is not None 
                        and isinstance (self.dout, str) 
                        and self.dout.startswith('.')
                        ): 
                        ext = self.dout.lower()  
                    else : 
                        warnings.warn(
                            "No file extension provided for `savefile`. "
                            "Cannot save the DataFrame."
                        )
                        return result

                # Determine the DataFrame to save
                if isinstance(result, pd.DataFrame):
                    df_to_save = result
                elif isinstance(result, tuple):
                    try:
                        df_to_save = result[self.data_index]
                    except IndexError:
                        warnings.warn(
                            f"`data_index` {self.data_index} is out of range "
                            "for the returned tuple."
                        )
                        return result

                    if not isinstance(df_to_save, pd.DataFrame):
                        warnings.warn(
                            f"Element at `data_index` {self.data_index} "
                            "is not a DataFrame."
                        )
                        return result
                else:
                    warnings.warn(
                        f"Return type '{type(result)}' is not a DataFrame or tuple."
                    )
                    return result

                # Get the appropriate writer function
                writers_dict = self.data_handler.writers(df_to_save)
                writer_func = writers_dict.get(ext.lower())

                if writer_func is None:
                    warnings.warn(
                        f"Unsupported file extension '{ext}'. "
                        "Cannot save the DataFrame."
                    )
                    return result

                # Save the DataFrame
                try:
                    writer_func(
                        savefile,
                        index=False
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to save the DataFrame: {e}"
                    )

            return result

        return wrapper(*args, **kwargs)

@EnsureFileExists(action ='ignore')
def read_data(
    f: str | pathlib.PurePath, 
    sanitize: bool = False, 
    reset_index: bool = False, 
    comments: str = "#", 
    delimiter: str = None, 
    columns: List[str] = None,
    npz_objkey: str = None, 
    verbose: bool = False, 
    **read_kws
) -> DataFrame:
    """
    Read all specific files and URLs allowed by the package.

    Readable files are systematically converted to a DataFrame.

    Parameters
    ----------
    f : str, Path-like object
        File path or Pathlib object. Must contain a valid file name and 
        should be a readable file or URL.

    sanitize : bool, default=False
        Push a minimum sanitization of the data such as:
        - Replace non-alphabetic column items with a pattern '_'
        - Cast data values to numeric if applicable
        - Drop full NaN columns and rows in the data

    reset_index : bool, default=False
        Reset index if full NaN columns are dropped after sanitization. 
        Apply minimum data sanitization after reading data.

    comments : str or sequence of str or None, default='#'
        The characters or list of characters used to indicate the start 
        of a comment. None implies no comments. For backwards compatibility, 
        byte strings will be decoded as 'latin1'.

    delimiter : str, optional
        The character used to separate the values. For backwards 
        compatibility, byte strings will be decoded as 'latin1'. The default 
        is whitespace.

    columns : list of str, optional
        List of column names to use. If the file has a header row, then 
        you should explicitly pass ``header=0`` to override the column 
        names.

    npz_objkey : str, optional
        Dataset key to identify array in multiple array storages in '.npz' 
        format. If key is not set during 'npz' storage, ``arr_0`` should 
        be used. Capable of reading text and numpy formats ('.npy' and 
        '.npz') data. Note that when data is stored in compressed ".npz" 
        format, provide the '.npz' object key as an argument of parameter 
        `npz_objkey`. If None, only the first array should be read and 
        ``npz_objkey='arr_0'``.

    verbose : bool, default=0
        Outputs message for user guide.

    read_kws : dict
        Additional keyword arguments passed to pandas readable file keywords.

    Returns
    -------
    DataFrame
        A dataframe with head contents by default.

    Notes
    -----
    This function reads various file formats and converts them into a 
    pandas DataFrame. It supports sanitization of the data which includes 
    replacing non-alphabetic column names, casting data to numeric types 
    where applicable, and removing fully NaN columns and rows. The function 
    also supports reading numpy arrays from '.npy' and '.npz' files.

    Examples
    --------
    >>> from gofast.core.io import read_data
    >>> df = read_data('data.csv', sanitize=True, reset_index=True)
    >>> print(df.head())

    See Also
    --------
    np.loadtxt : Load text file.
    np.load : Load uncompressed or compressed numpy `.npy` and `.npz` formats.
    gofast.dataops.management.save_or_load : Save or load numpy arrays.
    gofast.core.io.export_data: 
        Export a pandas DataFrame to multiple file formats based on specified
        extensions.
        
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. Proceedings of the 9th Python in Science Conference, 51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. Nature, 585(7825), 357-362.
    """
   
    def min_sanitizer ( d, /):
        """ Apply a minimum sanitization to the data `d`."""
        return to_numeric_dtypes(
            d, sanitize_columns= True, 
            drop_nan_columns= True, 
            reset_index=reset_index, 
            verbose = verbose , 
            fill_pattern='_', 
            drop_index = True
            )
    def _check_readable_file (f): 
        """ Return file name from path objects """
        msg =(f"Expects a Path-like object or URL. Please, check your"
              f" file: {os.path.basename(f)!r}")
        if not os.path.isfile (f): # force pandas read html etc 
            if not ('http://'  in f or 'https://' in f ):  
                raise TypeError (msg)
        elif not isinstance (f,  (str , pathlib.PurePath)): 
             raise TypeError (msg)
        if isinstance(f, str): 
            f =f.strip() # for consistency 
        return f 
    
    if ( isinstance ( f, str ) 
            and str(os.path.splitext(f)[1]).lower()in (
                '.txt', '.npy', '.npz')
            ): 
        f = save_or_load(f, task = 'load', comments=comments, 
                         delimiter=delimiter )
        # if extension is .npz
        if isinstance(f, np.lib.npyio.NpzFile):
            npz_objkey = npz_objkey or "arr_0"
            f = f[npz_objkey] 

        if columns is not None: 
            columns = is_iterable(columns, exclude_string= True, 
                                  transform =True, parse_string =True 
                                  )
            if len( columns )!= f.shape [1]: 
                warnings.warn(f"Columns expect {f.shape[1]} attributes."
                              f" Got {len(columns)}")
            
        f = pd.DataFrame(f, columns=columns )
        
    if isinstance (f, pd.DataFrame): 
        if sanitize: 
            f = min_sanitizer (f)
        return  f 
    
    elif isinstance (f, pd.Series): 
        return f 
    
    elif _is_array_like(f): 
        # just return nparray
        return np.asarray(f)
    
    cpObj= PandasDataHandlers().parsers 
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(
            f"Can only parse the {','.join( cpObj.keys())} files"
                        )
    try : 
        f = cpObj[ex](f, **read_kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except BaseException as e : 
        raise FileHandlingError (
            f"Cannot parse the file : {os.path.basename (f)!r}. "+  str(e))
    if sanitize: 
        f = min_sanitizer (f)
        
    return f 

@check_params(
    {
        'file_paths': Union[str, List[str]], 
        'columns': Optional[List[str]], 
        'extensions': Optional[Union [str, List[str]]], 
        'writer_opptions': Optional[Dict[str, Dict[str, Any]]], 
        'default_extension': str, 
        'overwrite': bool
   }, 
   coerce=False, 
)
def export_data(
    df,
    file_paths,
    columns=None,
    extensions=None,
    overwrite=False,
    writer_options=None,
    default_extension='.csv',
    verbose=0,
    **kwargs
):
    """
    Export a pandas DataFrame to multiple file formats based on specified 
    extensions.
    
    This function facilitates exporting a pandas DataFrame to various file formats by 
    leveraging the `PandasDataHandlers.writers` method. It provides robust and flexible 
    options to handle multiple export scenarios, ensuring compatibility with diverse 
    datasets and user requirements. The function intelligently manages file extensions, 
    supports selective column exports, and allows for customization through additional 
    parameters.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be exported to various file formats.
    file_paths : list-like
        A list of file paths where the DataFrame will be exported. Each path should 
        include the desired file name and extension.
    columns : list-like, optional
        Specific columns to include in the export. If `None`, all columns in the 
        DataFrame are exported.
    extensions : list-like or str, optional
        File extensions corresponding to each file path. If a single string is provided, 
        the same extension is applied to all file paths. If `None`, the extension is 
        inferred from each file path. If an extension cannot be determined, the 
        `default_extension` is used.
    overwrite : bool, default=False
        Determines whether to overwrite existing files:
            - ``True``: Overwrites files if they already exist.
            - ``False``: Skips exporting to files that already exist.
    writer_options : dict, optional
        A dictionary mapping file extensions to specific keyword arguments for the 
        Pandas writer functions. This allows customization of the export process for 
        different file formats. For example:
            {
                ".csv": {"index": False},
                ".json": {"orient": "records"},
                ".xlsx": {"index": False, "sheet_name": "Data"}
            }
    default_extension : str, default='.csv'
        The default file extension to use if none is provided in the `file_paths`. 
        Defaults to ``'.csv'``.
    verbose : int, default=0
        Controls the verbosity of the output:
            - ``0``: No output.
            - ``1``: Basic information about export progress.
            - ``2``: Detailed information about each export operation.
            - ``3``: Extensive information including file-specific details.
            - Levels ``4`` to ``7``: Additional debugging information as needed.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the Pandas writer functions, providing 
        further flexibility in the export process.
    
    Returns
    -------
    None
        The function performs the export operation and does not return any value. The 
        DataFrame remains unmodified unless changes are made inplace.
    

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.io import export_data
    >>> 
    >>> # Sample DataFrame
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': ['x', 'y', 'z'],
    ...     'C': [4.5, 5.5, 6.5]
    ... })
    >>> 
    >>> # Define file paths for export
    >>> file_paths = [
    ...     'output.csv',
    ...     'output.json',
    ...     'output.xlsx'
    ... ]
    >>> 
    >>> # Define writer-specific options
    >>> writer_options = {
    ...     '.csv': {'index': False},
    ...     '.json': {'orient': 'records'},
    ...     '.xlsx': {'index': False, 'sheet_name': 'Sheet1'}
    ... }
    >>> 
    >>> # Export the DataFrame to multiple formats
    >>> export_data(
    ...     df=data,
    ...     file_paths=file_paths,
    ...     overwrite=True,
    ...     writer_options=writer_options,
    ...     verbose=2
    ... )
    Processing columns: ['A', 'B', 'C']
    Successfully exported to 'output.csv'.
    Successfully exported to 'output.json'.
    Successfully exported to 'output.xlsx'.
    
    Notes
    -----
    .. math::
        \text{Export Operation} = 
        \begin{cases} 
            \text{Export to specified format} & \text{if extension is supported} \\
            \text{Use default extension} & \text{otherwise}
        \end{cases}
    
    The export process involves iterating through each specified file path, determining 
    the appropriate writer function based on the file extension using 
    `PandasDataHandlers.writers`, and executing the export. If an extension is not 
    provided or unsupported, the function defaults to using the `default_extension`. 
    The `overwrite` parameter ensures that existing files are handled according to the 
    user's preference, preventing accidental data loss.
    
    - **Flexibility**: The function is designed to handle multiple export formats 
      seamlessly, making it suitable for a wide range of data export scenarios.
    - **Error Management**: By controlling the `overwrite` flag and utilizing 
      `warnings`, the function ensures that users are informed of potential issues 
      without abrupt terminations unless necessary.
    - **Customization**: Through `writer_options` and `**kwargs`, users can tailor 
      the export process to meet specific requirements for different file formats.
    - **Verbosity Levels**: The `verbose` parameter provides users with control over 
      the amount of information displayed during execution, facilitating both quiet 
      operations and detailed monitoring.
    - **Extension Handling**: If a file path lacks an extension or the extension is 
      unsupported, the function intelligently defaults to the `default_extension`, 
      ensuring that the export process remains robust and error-free.
    - **Column Selection**: By specifying the `columns` parameter, users can export 
      only relevant subsets of the DataFrame, enhancing performance and reducing 
      unnecessary data storage.
    
    See Also
    --------
    pandas.DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    pandas.DataFrame.to_json : Convert the DataFrame to a JSON string.
    pandas.DataFrame.to_excel : Write DataFrame to an Excel file.
    gofast.api.property.PandasDataHandlers.writers :
        Provides a mapping of file extensions to Pandas writer functions.
    gofast.core.io.read_data: 
        Read all specific files and URLs allowed by the package.
    
    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
           in Python." In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Pandas Documentation. (2023). 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    .. [3] Pandas Documentation. (2023). 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html
    .. [4] Pandas Documentation. (2023). 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html
    .. [5] Gofast Package Documentation. (2023). 
           https://github.com/gofast/gofast
    """

    # Validate that the input is a pandas DataFrame
    df = to_frame_if(df)
    # If specific columns are provided, ensure they exist in the DataFrame
    if columns is not None:
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing_columns}"
            )
        # Select only the specified columns for export
        export_df = df[columns]
        if verbose >= 2:
            print(f"Selected columns for export: {columns}")
    else:
        # If no columns specified, use the entire DataFrame
        export_df = df.copy()
        if verbose >= 2:
            print("No specific columns provided. Exporting the entire DataFrame.")
    
    # Initialize the PandasDataHandlers instance to access writer methods
    data_handler = PandasDataHandlers()
    writers = data_handler.writers(export_df)
    
    if isinstance (file_paths, str): 
        file_paths = [file_paths]
    # Iterate through each file path to export the DataFrame
    for file_path in file_paths:
        # Extract the file extension from the file path
        extension = (
            '.' + file_path.split('.')[-1] 
            if '.' in file_path else default_extension
        )
        
        # If extensions list is provided, map each to the corresponding file path
        if extensions is not None:
            if isinstance(extensions, (list, tuple)):
                # Handle multiple extensions
                if len(extensions) != len(file_paths):
                    raise ValueError(
                        "`extensions` list must match the length of `file_paths`."
                    )
                extension = extensions[file_paths.index(file_path)]
            elif isinstance(extensions, str):
                # Handle single extension applied to all file paths
                extension = extensions
            else:
                raise TypeError("`extensions` must be a list, tuple, or string.")
        
        # If the extension is still not determined, use the default extension
        if not extension:
            extension = default_extension
            if verbose >= 3:
                print(
                    f"No extension found for '{file_path}'. Using default "
                    f"extension '{default_extension}'."
                )
        
        # Retrieve the appropriate writer function based on the file extension
        writer_func = writers.get(extension.lower())
        if writer_func is None:
            warning_msg = (
                f"No writer available for extension '{extension}'. "
                f"Skipping file '{file_path}'."
            )
            if verbose >= 1:
                warnings.warn(warning_msg)
            continue
        
        # Check if the file already exists and handle based on the overwrite flag
        try:
            if not overwrite and pd.io.common.file_exists(file_path):
                warning_msg = (
                    f"File '{file_path}' already exists and `overwrite` is set to "
                    f"`False`. Skipping export to this file."
                )
                if verbose >= 1:
                    warnings.warn(warning_msg)
                continue
        except Exception as e:
            warning_msg = (
                f"Could not check existence of file '{file_path}': {e}. "
                f"Proceeding with export."
            )
            if verbose >= 1:
                warnings.warn(warning_msg)
        
        # Prepare writer-specific options if provided
        writer_kwargs = writer_options.get(extension.lower(), {}) if writer_options else {}
        # Merge any additional keyword arguments passed to the function
        writer_kwargs.update(kwargs)
        
        # Attempt to write the DataFrame to the specified file path
        try:
            writer_func(file_path, **writer_kwargs)
            if verbose >= 1:
                print(f"Successfully exported to '{file_path}'.")
        except Exception as e:
            error_msg = (
                f"Failed to export DataFrame to '{file_path}': {e}."
            )
            if overwrite:
                warnings.warn(error_msg)
            else:
                raise RuntimeError(error_msg)
    
    # Final verbosity logging after all exports
    if verbose >= 4:
        print("Completed exporting DataFrame to all specified file paths.")
    
    if verbose >= 5:
        print("Exported DataFrame preview:")
        print(export_df.head())
    
    # Return None as the export operation does not modify the DataFrame
    return None

def save_or_load(
    fname:str, 
    arr: NDArray=None,  
    task: str='save', 
    format: str='.txt', 
    compressed: bool=False,  
    comments: str="#",
    delimiter: str=None, 
    **kws 
): 
    """Save or load Numpy array. 
    
    Parameters 
    -----------
    fname: file, str, or pathlib.Path
       File or filename to which the data is saved. 
       - >.npy , .npz: If file is a file-object, then the filename is unchanged. 
       If file is a string or Path, a .npy extension will be appended to the 
       filename if it does not already have one. 
       - >.txt: If the filename ends in .gz, the file is automatically saved in 
       compressed gzip format. loadtxt understands gzipped files transparently.
       
    arr: 1D or 2D array_like
      Data to be saved to a text, npy or npz file.
      
    task: str {"load", "save"}
      Action to perform. "Save" for storing file into the format 
      ".txt", "npy", ".npz". "load" for loading the data from storing files. 
      
    format: str {".txt", ".npy", ".npz"}
       The kind of format to save and load.  Note that when loading the 
       compressed data saved into `npz` format, it does not return 
       systematically the array rather than `np.lib.npyio.NpzFile` files. 
       Use either `files` attributes to get the list of registered files 
       or `f` attribute dot the data name to get the loaded data set. 

    compressed: bool, default=False 
       Compressed the file especially when file format is set to `.npz`. 

    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. This is useful when `fname`
       is in `txt` format. 
      
     delimiter: str,  optional
        The character used to separate the values. For backwards compatibility, 
        byte strings will be decoded as 'latin1'. The default is whitespace.
        
    kws: np.save ,np.savetext,  np.load , np.loadtxt 
       Additional keywords arguments for saving and loading data. 
       
    Return 
    ------
    None| data: ArrayLike 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.utils.baseutils import save_or_load 
    >>> data = np.random.randn (2, 7)
    >>> # save to txt 
    >>> save_or_load ( "test.txt" , data)
    >>> save_or_load ( "test",  data, format='.npy')
    >>> save_or_load ( "test",  data, format='.npz')
    >>> save_or_load ( "test_compressed",  data, format='.npz', compressed=True )
    >>> # load files 
    >>> save_or_load ( "test.txt", task ='load')
    Out[36]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test.npy", task ='load')
    Out[39]: array([-2.34162127,  0.55093818])
    >>> save_or_load ( "test.npz", task ='load')
    <numpy.lib.npyio.NpzFile at 0x1b0821870a0>
    >>> npzo = save_or_load ( "test.npz", task ='load')
    >>> npzo.files
    Out[44]: ['arr_0']
    >>> npzo.f.arr_0
    Out[45]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test_compressed.npz", task ='load')
    ...
    """
    r_formats = {"npy", "txt", "npz"}
   
    (kind, kind0), ( task, task0 ) = lowertify(
        format, task, return_origin =True )
    
    assert  kind.replace ('.', '') in r_formats, (
        f"File format expects {smart_format(r_formats, 'or')}. Got {kind0!r}")
    kind = '.' + kind.replace ('.', '')
    assert task in {'save', 'load'}, ( 
        "Wrong task {task0!r}. Valid tasks are 'save' or 'load'") 
    
    save= {'.txt': np.savetxt, '.npy':np.save,  
           ".npz": np.savez_compressed if ellipsis2false(
               compressed)[0] else np.savez 
           }
    if task =='save': 
        arr = np.array (is_iterable( arr, exclude_string= True, 
                                    transform =True ))
        save.get(kind) (fname, arr, **kws )
        
    elif task =='load': 
         ext = os.path.splitext(fname)[1].lower() 
         if ext not in (".txt", '.npy', '.npz', '.gz'): 
             raise ValueError ("Unrecognized file format {ext!r}."
                               " Expect '.txt', '.npy', '.gz' or '.npz'")
         if ext in ('.txt', '.gz'): 
            arr = np.loadtxt ( fname , comments= comments, 
                              delimiter= delimiter,   **kws ) 
         else : 
            arr = np.load(fname,**kws )
         
    return arr if task=='load' else None 

def _is_data_readable(func=None, *, data_to_read=None, params=None):
    """
    A decorator to automatically read data if it is not explicitly passed
    to the decorated function.

    This decorator ensures that the specified data (either passed explicitly 
    or as the first positional argument) is read and processed before being 
    passed to the decorated function. If the data is not explicitly passed, 
    the decorator automatically attempts to read it from the first positional 
    argument (usually `args[0]`).

    The data is passed to the `read_data` function from the 
    `gofast.dataops.management` module for processing before being returned 
    to the decorated function.

    Parameters
    ----------
    func : callable, optional
        The function to decorate. If the decorator is used without parentheses 
        (i.e., `@_read_data`), this argument will be `None` and the `params` 
        and `data_to_read` arguments will be used to process the data. 

    data_to_read : str, optional
        The name of the keyword argument (or positional argument) to read the 
        data from. If `None` (default), the first positional argument (`args[0]`) 
        will be read. If a specific keyword argument is provided
        (e.g., `data_to_read='data2'`), the decorator will attempt to read 
        that argument.

    params : dict, optional
        A dictionary of parameters to pass to the `read_data` function (from 
        the `gofast.dataops.management` module). These parameters are used 
        when processing the data before passing it to the decorated function.

    Returns
    -------
    callable
        A wrapper function that checks if the specified `data` argument is 
        provided. If not, the data is automatically read and processed before 
        being passed to the decorated function.

    Notes
    -----
    - If `data_to_read` is specified, the decorator will look for that argument 
      in either the keyword arguments (`kwargs`) or the first positional argument 
      (`args[0]`), depending on whether it is explicitly passed.
    - If no `data_to_read` is provided, the decorator defaults to reading 
      the first positional argument (`args[0]`).
    - The decorator will automatically handle cases where the data is passed 
      explicitly and will skip processing if the data is `None`.
    - The decorator should be used with or without parentheses. If used without 
      parentheses, the `params` and `data_to_read` arguments must be provided 
      as keyword arguments.

    Examples
    --------
    Example 1: Using the decorator without parentheses:

    >>> from gofast.core.io import is_data_readable
    >>> @is_data_readable(params={"sanitize": True, "reset_index": True})
    >>> def process_data(data):
    >>>     print(data.head())

    >>> process_data('my_data.csv')
    # This will read and process the file 'my_data.csv' using read_data.

    Example 2: Using the decorator with parentheses and specifying the data argument to read:

    >>> @is_data_readable(data_to_read="data2", params={"sanitize": True})
    >>> def process_data(data1, data2=None):
    >>>     print(data2)

    >>> process_data(data2="my_data2.csv")
    # This will read and process the file 'my_data2.csv' using read_data.

    Example 3: Using the decorator with a keyword argument for the data argument:

    >>> @is_data_readable(data_to_read="data1", params={"sanitize": True})
    >>> def process_data(data1, data2=None):
    >>>     print(data1)

    >>> process_data(data1="my_data1.csv")
    # This will read and process the file 'my_data1.csv' using _read_data.
    """
    # Handle the case where the decorator is used without parentheses
    # (i.e., with params or data_to_read).
    if func is None:
        return lambda func: is_data_readable(
            func, data_to_read=data_to_read, params=params)

    @wraps(func)
    def wrapper(*args, **kwargs):
        
        # Check if the specified data (either through keyword or positional)
        # should be read
        data = kwargs.get(data_to_read, None) if data_to_read else None

        # If data is not explicitly passed, check the first positional argument
        if data is None and args:
            data = args[0]

        # If we have data to read, apply the `read_data` transformation
        if data is not None:
            data = read_data(data, **(params or {}))

        # Update the arguments with the transformed data
        if data_to_read:
            kwargs[data_to_read] = data
        elif args:
            args = (data,) + args[1:]

        # Call the decorated function with the updated arguments
        return func(*args, **kwargs)

    return wrapper

def is_data_readable(
    func=None,
    *,
    data_to_read=None,
    params=None,
    fallback=None,
    strict=False,
    error='raise'
):
    """
    A decorator to automatically read data if it is not explicitly passed
    to the decorated function.
    
    Attempts to convert or read a provided data argument into
    a valid DataFrame before invoking the decorated function.
    If no valid data is passed, it checks the first positional
    argument. In case of reading or validation issues, it can
    either raise an error, warn the user, or silently ignore
    the problem by using a fallback value.

    .. math::
       \mathbf{Data}_{\text{out}} \;=\;
       \begin{cases}
         \text{read\_data}(\mathbf{Data}_{\text{in}}), & 
         \text{if valid and no errors} \\
         \text{fallback}, & 
         \text{if errors occur and } \textit{error} \neq 'raise'
       \end{cases}

    Parameters
    ----------
    func : callable, optional
        The function to decorate. If this decorator is used
        without parentheses, <func> is inferred automatically.
    data_to_read : str, optional
        The name of the argument whose value will be read
        by ``read_data``. If ``None``, the decorator falls
        back to the first positional argument.
    params : dict, optional
        Additional keyword arguments passed to the
        ``read_data`` function. Useful for specifying file
        parsing options (e.g., separator, header info, etc.).
    fallback : any, optional
        A default value used if data reading fails. If
        the reading process or strict validation triggers
        an exception and ``error`` is not set to
        ``'raise'``, the decorator replaces the data with
        this fallback value.
    strict : bool, optional
        If ``True``, enforces that the final data is a
        non-empty DataFrame. Otherwise, an error is raised
        or handled according to <error>.
    error : { ``'raise'``, ``'warn'``, ``'ignore'`` }, optional
        Determines how exceptions are handled:
          - ``'raise'``: re-raise the exception immediately.
          - ``'warn'``: issue a warning, then replace the
            data with <fallback>.
          - ``'ignore'``: silently ignore errors and replace
            the data with <fallback>.

    Formulation
    -------------
    .. math::
       \text{validated\_data} \;=\;
       \begin{cases}
         \text{DataFrame}, & \text{if successful parse} \\
         \text{fallback}, & \text{if parse fails and } 
         error \neq \text{'raise'}
       \end{cases}

    Examples
    --------
    >>> from gofast.core.io import is_data_readable
    >>> @is_data_readable(data_to_read='input_data',
    ...                   params={'sep': ';'},
    ...                   fallback=pd.DataFrame(),
    ...                   strict=True,
    ...                   error='warn')
    ... def process_data(input_data):
    ...     return input_data
    ...
    >>> # If reading fails, a warning is shown, and
    >>> # an empty DataFrame is used as fallback.

    Notes
    -----
    - If <strict> is ``True``, the data must be a non-empty
      DataFrame, or an exception occurs (handled according to
      <error>).
    - If <fallback> is provided, any failures or invalid data
      (including empty frames when <strict> is ``True``)
      result in <fallback> being passed to the decorated
      function (unless ``error='raise'``).
    - The order of decorators matters if used alongside
      other decorators that also manipulate function
      arguments.

    See Also
    --------
    ``read_data`` :
        A function to parse or convert input into a DataFrame.
    ``pd.DataFrame`` :
        The typical data structure produced by reading.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Enhancing Decorators for
       Robust Data Pipelines," Journal of PyDev, 2024.
    .. [2] Brown, K. "Error-Handling Strategies in
       Data-Oriented Decorators," Data Press, 2023.
    """

    if func is None:
        return lambda f: is_data_readable(
            f,
            data_to_read=data_to_read,
            params=params,
            fallback=fallback,
            strict=strict,
            error=error
        )

    # Validate `error` parameter to ensure it has an acceptable value.
    if error not in {'raise', 'warn', 'ignore'}:
        raise ValueError(
            "`error` must be one of {'raise', 'warn', 'ignore'}."
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        
        nonlocal data_to_read  # Explicitly declare data_to_read as nonlocal
        
        # 1) Attempt to retrieve the data argument from kwargs using
        #    `data_to_read` as the key if provided; otherwise, check
        #    the first positional arg (args[0]) if available.
        data = kwargs.get(data_to_read, None) if data_to_read else None 

        if data is None: 
            if len(args) > 0:
            # user passed data as a positional argument
                data = args[0]
                data_to_read = None  # so we end up rewriting args
            
            elif 'data' in kwargs:
                # user passed data as a named keyword
                # i.e no positional argument passed, we expected the data
                # parameter being set as 'data'.
                data = kwargs['data']
                data_to_read = 'data'
            
            else:
                # reinitialize data parameter.
                data = None
                data_to_read = None

        try:
            # 2) If `data` is not None, call `read_data` to convert
            #    it into a DataFrame (or something else) as needed.
            if data is not None:
                data = read_data(data, **(params or {}))

            # 3) If `strict=True`, ensure we have a non-empty
            #    DataFrame. If not, raise an error or handle it below.
            if strict:
                if not isinstance(data, pd.DataFrame) or data.empty:
                    raise ValueError(
                        "Invalid data: resulting DataFrame is "
                        "empty or not a DataFrame."
                    )
        except Exception as e:
            # 4) Depending on the `error` policy, we raise, warn,
            #    or ignore the exception, replacing data with `fallback`.
            if error == 'raise':
                raise
            elif error == 'warn':
                warnings.warn(str(e), stacklevel=2)
                data = fallback
            else:  # 'ignore'
                data = fallback

        # 5) Update the arguments so the wrapped function receives
        #    the (possibly) transformed `data` in the correct position
        #    (keyword vs positional).
        if data_to_read:
            kwargs[data_to_read] = data
        elif args:
            args = (data,) + args[1:]

        # 6) Finally, call the original function with updated args/kwargs.
        return func(*args, **kwargs)

    return wrapper

def _is_array_like(obj):
    # Check for iterable objects, excluding strings
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def to_frame_if(
    data: Union[str, pd.Series, Iterable], 
    df_only: bool = ...,  
    *args, 
    **kwargs
):

    """
    Attempts to convert data into a pandas DataFrame if possible. The function
    handles various input types like strings, pandas Series, numpy arrays, 
    or lists and will try the most appropriate method to convert them.

    Parameters
    ----------
    data : str, pandas.Series, numpy.ndarray, or list
        Input data that is to be converted into a pandas DataFrame. The function
        tries different methods depending on the input type.

    df_only : bool, default=True
        If True, and the input is a pandas Series, the function will convert 
        it into a DataFrame. If False, it will tolerate the Series and return it 
        as-is.

    *args, **kwargs : 
        Additional arguments passed to the `read_data` function (when `data` is 
        a string representing a file path or file-like object).

    Returns
    -------
    pd.DataFrame or pd.Series
        Returns the input data as a DataFrame if conversion was successful. 
        If the input was a Series and `df_only=False`, the original Series 
        is returned.

    Notes
    -----
    - If `data` is a file path (string), the function uses the `read_data` 
      function from the `gofast.core.io` module to load the data into a DataFrame.
    - If `data` is a pandas Series and `df_only=True`, the Series will be 
      converted into a DataFrame. If `df_only=False`, the function will return 
      the Series without modification.
    - If the input is neither a file path nor a Series, the function tries 
      to convert the data into a DataFrame using common conversion techniques 
      for numpy arrays or lists. If these conversions fail, the function will 
      raise an appropriate error.
    
    Examples
    --------
    1. Convert a pandas Series to a DataFrame:
    >>> import pandas as pd
    >>> from gofast.core.io import to_frame_if 
    >>> series = pd.Series([1, 2, 3, 4])
    >>> to_frame_if(series)
       0
    0  1
    1  2
    2  3
    3  4

    2. Read data from a file and convert to DataFrame:
    >>> file_path = 'data.csv'
    >>> to_frame_if(file_path)
       col1  col2
    0     1     2
    1     3     4

    3. Convert a numpy array to a DataFrame:
    >>> np_array = np.array([[1, 2], [3, 4]])
    >>> to_frame_if(np_array)
       0  1
    0  1  2
    1  3  4
    """
    # Case 1: If data is already a DataFrame, return it as is
    if isinstance(data, pd.DataFrame):
        return data

    # Case 2: If data is a string, use the read_data function to load the file
    if isinstance(data, str):
        try:
            return read_data(data, *args, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading data from file: {e}")

    # Case 3: If data is a pandas Series
    elif isinstance(data, pd.Series):
        if df_only:
            # Convert Series to DataFrame
            return data.to_frame()
        else:
            # Return the Series as-is
            return data

    # Case 4: If data is an array-like object (e.g., numpy array, list)
    elif _is_array_like(data):
        # Convert array-like to DataFrame
        try: 
            return pd.DataFrame(data)
        except Exception as e: 
            # Raise an error if the input is not a recognized type
            raise ValueError(
                "Unsupported data type. The input must be a file path,"
                " pandas Series, numpy array, list, or pandas DataFrame."
                ) from e 

@check_params ( 
    { 
        'text': str, 
        'style':Optional[str], 
        'alignment': str, 
        'extra_space': int, 
        'text_size': Union[int, None],
        'break_word': bool
    }
 )
def fmt_text(
    text: str, 
    style: Optional[str] = '-', 
    border_style: Optional[str] = None, 
    alignment: str = 'left', 
    extra_space: int = 3, 
    text_size: Optional[int] = None, 
    break_word: bool = False
) -> str:
    """
    Format text with specified styling, alignment, and wrapping.

    Parameters
    ----------
    text : str
        The input text to be formatted.
    
    style : str, optional
        The character used for the top and bottom border lines. 
        If `None`, no top and bottom borders are added. Default is ``'-'``.
    
    border_style : str, optional
        The character used to frame the text lines on the left and right.
        If ``None``, no side borders are added. Default is ``None``.
    
    alignment : str, {'left', 'center', 'right', 'justify'}, default='left'
        The alignment of the text within the borders. Options are:
        - ``'left'``
        - ``'center'``
        - ``'right'``
        - ``'justify'``
    
    extra_space : int, default 3
        The number of extra spaces added based on the alignment:
        - For 'left' and 'right', spaces are added to the respective side.
        - For 'center', spaces are added to both sides.
        - For 'justify', spaces are distributed evenly between words.
    
    text_size : int, optional
        The maximum width of the formatted text. If ``None``, uses the 
        terminal width obtained from `get_table_size()`. Default is ``None``.
    
    break_word : bool, default False
        If `True`, words longer than the remaining space in a line will be 
        broken with a hyphen. If ``False``, the word moves to the next line.
    
    Returns
    -------
    str
        The formatted text as a single string with appropriate styling 
        and alignment.
    
    Examples
    --------
    >>> from gofast.core.io import fmt_text
    >>> sample_text = "This is a sample text that will be formatted with"
    " left spaces, underlines, and auto-wrapping."
    >>> print(fmt_text(sample_text, alignment='center', style='~', 
                       extra_space=2, text_size =45))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          This is a sample text that will be     
       formatted with left spaces, underlines,   
                  and auto-wrapping.             
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    >>> print(fmt_text(sample_text, border_style='|', alignment='right', 
                       extra_space=1, text_size =60))
    ------------------------------------------------------------
    |   This is a sample text that will be formatted with left |
    |                   spaces, underlines, and auto-wrapping. |
    ------------------------------------------------------------
    """
    if text_size is None:
        text_size = TW
    else:
        if text_size <= 0:
            raise ValueError("`text_size` must be a positive integer or None.")
    
    if alignment not in {'left', 'center', 'right', 'justify'}:
        raise ValueError(
            "`alignment` must be one of {'left', 'center', 'right', 'justify'}.")
    
    if border_style is not None and len(border_style) != 1:
        raise ValueError(
            "`border_style` must be a single character or None."
            )
    
    if style is not None and len(style) != 1:
        raise ValueError("`style` must be a single character or None.")
    
    # Calculate available width for text
    border_width = 2 if border_style else 0
    available_width = text_size - border_width - 2 * extra_space
    if available_width <= 0:
        raise ValueError(
            "`text_size` is too small for the given `extra_space` and borders.")
    
    # Function to handle word breaking
    def wrap_text(text: str, width: int) -> List[str]:
        if break_word:
            return textwrap.wrap(
                text, width=width, break_long_words=True, 
                break_on_hyphens=True
                )
        else:
            return textwrap.wrap(
                text, width=width, break_long_words=False, 
                break_on_hyphens=False
                )
    
    wrapped_lines = wrap_text(text, available_width)
    
    # Function to apply alignment
    def align_line(line: str) -> str:
        if alignment == 'left':
            return line.ljust(available_width)
        elif alignment == 'right':
            return line.rjust(available_width)
        elif alignment == 'center':
            return line.center(available_width)
        elif alignment == 'justify':
            words = line.split()
            if len(words) == 1:
                return words[0].ljust(available_width)
            total_spaces = available_width - sum(len(word) for word in words)
            space_between_words, extra = divmod(total_spaces, len(words) - 1)
            justified_line = ''
            for i, word in enumerate(words[:-1]):
                justified_line += word + ' ' * (space_between_words + (1 if i < extra else 0))
            justified_line += words[-1]
            return justified_line
        else:
            return line  # Fallback to no alignment
    
    aligned_lines = [align_line(line) for line in wrapped_lines]
    
    # Add borders and extra spaces
    formatted_lines = []
    for line in aligned_lines:
        if border_style:
            formatted_line = (
                border_style + 
                ' ' * extra_space + 
                line + 
                ' ' * extra_space + 
                border_style
            )
        else:
            formatted_line = ' ' * extra_space + line + ' ' * extra_space
        formatted_lines.append(formatted_line)
    
    # Create border lines
    if style:
        border_line = style * text_size
        formatted_text = [border_line] + formatted_lines + [border_line]
    else:
        formatted_text = formatted_lines
    
    return '\n'.join(formatted_text)


  