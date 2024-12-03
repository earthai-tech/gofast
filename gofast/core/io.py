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
import numpy as np 
import pandas as pd 
from collections.abc import Iterable
from functools import wraps 
from typing import List, Union, Optional, Callable  

from ..exceptions import FileHandlingError 
from ..api.types import DataFrame, NDArray
from ..api.property import PandasDataHandlers
from .checks import is_iterable 
from .array_manager import to_numeric_dtypes 
from .utils import  ellipsis2false, lowertify, smart_format 

__all__=[
    "EnsureFileExists", 
    "read_data",
    "save_or_load",
    "is_data_readable",
    "to_frame_if"
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
    >>> from gofast.tools.baseutils import save_or_load 
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

def is_data_readable(func=None, *, data_to_read=None, params=None):
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


  