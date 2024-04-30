# -*- coding: utf-8 -*-
from __future__ import annotations 
import os 
import copy 
import time
import shutil
import inspect
import pathlib
import warnings
import functools 
import threading 
import subprocess 
from joblib import Parallel, delayed
from datetime import datetime
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.ticker import FixedLocator
from sklearn.utils import all_estimators 

from ..api.property import  Config
from ..api.types import Union, List, Optional, Tuple, Iterable, Any, Set 
from ..api.types import _T, _F, DataFrame, ArrayLike, Series, NDArray
from ..decorators import Dataify 
from ..exceptions import FileHandlingError
from .coreutils import is_iterable , ellipsis2false, smart_format  
from .coreutils import to_numeric_dtypes, validate_feature
from .coreutils import _assert_all_types, exist_features 
from .validator import check_consistent_length, get_estimator_name
from .validator import _is_arraylike_1d, array_to_frame, build_data_if
from .validator import is_categorical 
from ._dependency import import_optional_dependency

def smart_rotation(ax):
    """
    Automatically adjusts the rotation of x-axis tick labels on a matplotlib
    axis object based on the overlap of labels. This function assesses the
    overlap by comparing the horizontal extents of adjacent tick labels. If
    any overlap is detected, it rotates the labels by 45 degrees to reduce
    or eliminate the overlap. If no overlap is detected, labels remain
    horizontal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object for which to adjust the tick label rotation.

    Examples
    --------
    # Example of creating a simple time series plot with date overlap handling
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.dates import DateFormatter
    >>> from gofast.tools.baseutils import smart_rotation

    # Generate a date range and some random data
    >>> dates = pd.date_range(start="2020-01-01", periods=100, freq='D')
    >>> values = np.random.rand(100)

    # Create a DataFrame
    >>> df = pd.DataFrame({'Date': dates, 'Value': values})

    # Create a plot
    >>> fig, ax = plt.subplots()
    >>> ax.plot(df['Date'], df['Value'])
    >>> ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # Apply smart rotation to adjust tick labels dynamically
    >>> smart_rotation(ax)

    # Show the plot
    >>> plt.show()

    Notes
    -----
    This function needs to be used in conjunction with matplotlib plots where
    the axis ('ax') is already set up with tick labels. It is especially useful
    in time series and other plots where the x-axis labels are dates or other
    large strings that may easily overlap. Drawing the canvas (plt.gcf().canvas.draw())
    is necessary to render the labels and calculate their positions, which may
    impact performance for very large plots or in tight loops.
    
    """
    
    # Draw the canvas to get the labels rendered, which is necessary for calculating overlap
    plt.gcf().canvas.draw()

    # Retrieve the x-axis tick labels and their extents
    labels = [label.get_text() for label in ax.get_xticklabels()]
    tick_locs = ax.get_xticks()  # get the locations of the current ticks
    label_extents = [label.get_window_extent() for label in ax.get_xticklabels()]

    # Check for overlap by examining the extents
    overlap = False
    num_labels = len(label_extents)
    for i in range(num_labels - 1):
        if label_extents[i].xmax > label_extents[i + 1].xmin:
            overlap = True
            break

    # Apply rotation if overlap is detected
    rotation = 45 if overlap else 0

    # Set the locator before setting labels
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.set_xticklabels(labels, rotation=rotation)
    
def select_features(
    data: DataFrame,
    features: Optional[List[str]] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    coerce: bool = False,
    columns: Optional[List[str]] = None,
    verify_integrity: bool = False,
    parse_features: bool = False,
    **kwd
) -> DataFrame:
    """
    Selects features from a DataFrame according to specified criteria,
    returning a new DataFrame.

    Parameters
    ----------
    data : DataFrame
        The DataFrame from which to select features.
    features : List[str], optional
        Specific feature names to select. An error is raised if any
        feature is not present in `data`.
    include : str or List[str], optional
        The data type(s) to include in the selection. Possible values
        are the same as for the pandas `include` parameter in `select_dtypes`.
    exclude : str or List[str], optional
        The data type(s) to exclude from the selection. Possible values
        are the same as for the pandas `exclude` parameter in `select_dtypes`.
    coerce : bool, default False
        If True, numeric columns are coerced to the appropriate types without
        selection, ignoring `features`, `include`, and `exclude` parameters.
    columns : List[str], optional
        Columns to construct a DataFrame if `data` is passed as a Numpy array.
    verify_integrity : bool, default False
        Verifies the data type integrity and converts data to the correct
        types if necessary.
    parse_features : bool, default False
        Parses string features and converts them to an iterable object.
    **kwd : dict
        Additional keyword arguments for `pandas.DataFrame.astype`.

    Returns
    -------
    DataFrame
        A new DataFrame with the selected features.

    Examples
    --------
    >>> from gofast.tools.baseutils import select_features
    >>> data = {"Color": ['Blue', 'Red', 'Green'],
                "Name": ['Mary', "Daniel", "Augustine"],
                "Price ($)": ['200', "300", "100"]}
    >>> select_features(data, include='number')
    Empty DataFrame
    Columns: []
    Index: [0, 1, 2]

    >>> select_features(data, include='number', verify_integrity=True)
        Price ($)
    0   200.0
    1   300.0
    2   100.0

    >>> select_features(data, features=['Color', 'Price ($)'])
       Color Price ($)
    0  Blue  200
    1  Red   300
    2  Green 100

    See Also
    --------
    pandas.DataFrame.select_dtypes : For more information on how to use
                                      `include` and `exclude`.

    Reference
    ---------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    
    """
    coerce, verify_integrity, parse_features= ellipsis2false( 
        coerce, verify_integrity, parse_features)
    
    data = build_data_if(data, columns = columns, )
  
    if verify_integrity: 
        data = to_numeric_dtypes(data )
        
    if features is not None: 
        features= list(is_iterable (
            features, exclude_string=True, transform=True, 
            parse_string = parse_features)
            )
        validate_feature(data, features, verbose ='raise')
    # change the dataype 
    data = data.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return data [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return data if coerce else data.select_dtypes (include, exclude) 

def speed_rowwise_process(
    data, /, 
    func, 
    n_jobs=-1
    ):
    """
    Processes a large dataset by applying a complex function to each row. 
    
    Function utilizes parallel processing to optimize for speed.

    Parameters
    ----------
    data : pd.DataFrames
        The large dataset to be processed. Assumes the 
        dataset is a Pandas DataFrame.

    func : function
        A complex function to apply to each row of the dataset. 
        This function should take a row of the DataFrame as 
        input and return a processed row.

    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using 
        all processors. Default is -1.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Example
    -------
    >>> def complex_calculation(row):
    >>>     # Example of a complex row-wise calculation
    >>>     return row * 2  # This is a simple placeholder for demonstration.
    >>>
    >>> large_data = pd.DataFrame(np.random.rand(10000, 10))
    >>> processed_data = speed_rowwise_process(large_data, complex_calculation)

    """
    # Function to apply `func` to each row in parallel
    def process_row(row):
        return func(row)

    # Using Joblib's Parallel and delayed to apply the function in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_row)(row) 
                                      for row in data.itertuples(index=False))

    # Converting results back to DataFrame
    processed_data = pd.DataFrame(results, columns=data.columns)
    return processed_data
    
def run_shell_command(command, progress_bar_duration=30, pkg=None):
    """
    Run a shell command with an indeterminate progress bar.

    This function will display a progress bar for a predefined duration while 
    the package installation command runs in a separate thread. The progress 
    bar is purely for visual effect and does not reflect the actual 
    progress of the installation.

    Keep in mind:
    
    This function assumes that you have tqdm installed (pip install tqdm).
    The actual progress of the installation isn't tracked; the progress bar 
    is merely for aesthetics.
    The function assumes the command is a blocking one 
    (like most pip install commands) and waits for it to complete.
    Adjust progress_bar_duration based on how long you expect the installation
    to take. If the installation finishes before the progress bar, the bar
    will stop early. If the installation takes longer, the bar will complete, 
    but the function will continue to wait until the installation is done.
    
    Parameters:
    -----------
    command : list
        The command to run, provided as a list of strings.

    progress_bar_duration : int
        The maximum duration to display the progress bar for, in seconds.
        Defaults to 30 seconds.
    pkg: str, optional 
        The name of package to install for customizing bar description. 

    Returns:
    --------
    None
    
    Example 
    -------
    >>> from gofast.tools.baseutils import run_shell_command 
    >>> run_shell_command(["pip", "install", "gofast"])
    """
    def run_command(command):
        subprocess.run(command, check=True)

    def show_progress_bar(duration):
        with tqdm(total=duration, desc="Installing{}".format( 
                '' if pkg is None else f" {str(pkg)}"), 
                  bar_format="{l_bar}{bar}", ncols=77, ascii=True)  as pbar:
            for i in range(duration):
                time.sleep(1)
                pbar.update(1)

    # Start running the command
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()

    # Start the progress bar
    show_progress_bar(progress_bar_duration)

    # Wait for the command to finish
    thread.join()


def download_file(url, local_filename , dstpath =None ):
    """download a remote file. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
       
    Example 
    ---------
    >>> from gofast.tools.baseutils import download_file
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename, test_directory)    
    
    """
    import_optional_dependency("requests")
    import requests 
    print("{:-^70}".format(f" Please, Wait while {os.path.basename(local_filename)}"
                          " is downloading. ")) 
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    local_filename = os.path.join( os.getcwd(), local_filename) 
    
    if dstpath: 
         move_file ( local_filename,  dstpath)
         
    print("{:-^70}".format(" ok! "))
    
    return None if dstpath else local_filename


def fancier_downloader(url, local_filename, dstpath =None ):
    """ Download remote file with a bar progression. 
    
    Parameters 
    -----------
    url: str, 
      Url to where the file is stored. 
    loadl_filename: str,
      Name of the local file 
      
    dstpath: Optional 
      The destination path to save the downloaded file. 
      
    Return 
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise. 
    Example
    --------
    >>> from gofast.tools.baseutils import fancier_downloader
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename)

    """
    import_optional_dependency("requests")
    import requests 
    try : 
        from tqdm import tqdm
    except: 
        # if tqm is not install 
        return download_file (url, local_filename, dstpath  )
        
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Get the total file size from header
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', 
                            unit_scale=True, ncols=77, ascii=True)
        with open(local_filename, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
    local_filename = os.path.join( os.getcwd(), local_filename) 
    
    if dstpath: 
         move_file ( local_filename,  dstpath)
         
    return local_filename


def move_file(file_path, directory):
    """ Move file to a directory. 
    
    Create a directory if not exists. 
    
    Parameters 
    -----------
    file_path: str, 
       Path to the local file 
    directory: str, 
       Path to locate the directory.
    
    Example 
    ---------
    >>> from gofast.tools.baseutils import move_file
    >>> file_path = 'path/to/your/file.txt'  # Replace with your file's path
    >>> directory = 'path/to/your/directory'  # Replace with your directory's path
    >>> move_file(file_path, directory)
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Move the file to the directory
    shutil.move(file_path, os.path.join(directory, os.path.basename(file_path)))

def check_file_exists(package, resource):
    """
    Check if a file exists in a package's directory with 
    importlib.resources.

    :param package: The package containing the resource.
    :param resource: The resource (file) to check.
    :return: Boolean indicating if the resource exists.
    
    :example: 
        >>> from gofast.tools.baseutils import check_file_exists
        >>> package_name = 'gofast.datasets.data'  # Replace with your package name
        >>> file_name = 'h.h5'    # Replace with your file name

        >>> file_exists = check_file_exists(package_name, file_name)
        >>> print(f"File exists: {file_exists}")
    """

    import importlib.resources as pkg_resources
    return pkg_resources.is_resource(package, resource)

def is_readable (
        f:str, 
        *, 
        as_frame:bool=False, 
        columns:List[str]=None,
        input_name='f', 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a pandas frame.  
    
    Parameters 
    -----------
    f: Path-like object -Should be a readable files or url  
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the columns orthewise 
        no-action is performed and return the same array.
    input_name : str, default=""
        The data name used to construct the error message. 
        
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
        If ``ignore``, warnings silence mode is triggered.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
        
    force:bool, default=False
        Force conversion array to a frame is columns is not supplied.
        Use the combinaison, `input_name` and `X.shape[1]` range.
        
    kws: dict, 
        Pandas readableformats additional keywords arguments. 
    Returns
    ---------
    f: pandas dataframe 
         A dataframe with head contents... 
    
    """
    def _check_readable_file (f): 
        """ Return file name from path objects """
        msg =(f"Expects a Path-like object or URL. Please, check your"
              f" file: {os.path.basename(f)!r}")
        if not os.path.isfile (f): # force pandas read html etc 
            if not ('http://'  in f or 'https://' in f ):  
                raise TypeError (msg)
        elif not isinstance (f,  (str , pathlib.PurePath)): 
             raise TypeError (msg)
        if isinstance(f, str): f =f.strip() # for consistency 
        return f 
    
    if hasattr (f, '__array__' ) : 
        f = array_to_frame(
            f, 
            to_frame= True , 
            columns =columns, 
            input_name=input_name , 
            raise_exception= True, 
            force= True, 
            )
        return f 

    cpObj= Config().parsers 
    
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} "
                        f" files not {ex!r}.")
    try : 
        f = cpObj[ex](f, **kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 

def lowertify(
    *values,
    strip: bool = True, 
    return_origin: bool = False, 
    unpack: bool = False
    ) -> Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]:
    """
    Convert all input values to lowercase strings, optionally stripping 
    whitespace, and optionally return the original values alongside the 
    lowercased versions.
    
    Can also unpack the tuples of lowercased and original values into a single
    flat tuple.

    Parameters
    ----------
    *values : Any
        Arbitrary number of values to be converted to lowercase. Non-string 
        values will be converted to strings before processing.
    strip : bool, optional
        If True (default), leading and trailing whitespace will be removed 
        from the strings.
    return_origin : bool, optional
        If True, each lowercased string is returned as a tuple with its 
        original value; otherwise, only the lowercased strings are returned.
    unpack : bool, optional
        If True, and `return_origin` is also True, the function returns a 
        single flat tuple containing all lowercased and original values 
        alternatively. This parameter is ignored if `return_origin` is False.

    Returns
    -------
    Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]
        Depending on `return_origin` and `unpack` flags, returns either:
        - A tuple of lowercased (and optionally stripped) strings.
        - A tuple of tuples, each containing the lowercased string and its 
          original value.
        - A single flat tuple containing all lowercased and original values 
          alternatively (if `unpack` is True).

    Examples
    --------
    >>> from gofast.tools.baseutils import lowertify
    >>> lowertify('KIND')
    ('kind',)
    
    >>> lowertify("KIND", return_origin=True)
    (('kind', 'KIND'),)
    
    >>> lowertify("args1", 120, 'ArG3')
    ('args1', '120', 'arg3')
    
    >>> lowertify("args1", 120, 'ArG3', return_origin=True)
    (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
    
    >>> lowertify("KIND", "task ", return_origin=True, unpack=True)
    ('kind', 'KIND', 'task', 'task ')
    """
    processed_values = [(str(val).strip().lower() if strip 
                         else str(val).lower(), val) for val in values]
    if return_origin:
        if unpack:
            # Flatten the list of tuples into a single tuple for unpacking
            return tuple(item for pair in processed_values for item in pair)
        else:
            return tuple(processed_values)
    else:
        return tuple(lowered for lowered, _ in processed_values)


def save_or_load(
    fname:str, /,
    arr: NDArray=None,  
    task: str='save', 
    format: str='.txt', 
    compressed: bool=...,  
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

def array2hdf5 (
    filename: str, /, 
    arr: NDArray=None , 
    dataname: str='data',  
    task: str='store', 
    as_frame: bool =..., 
    columns: List[str]=None, 
)-> NDArray | DataFrame: 
    """ Load or write array to hdf5
    
    Parameters 
    -----------
    arr: Arraylike ( m_samples, n_features) 
      Data to load or write 
    filename: str, 
      Hdf5 disk file name whether to write or to load 
    task: str, {"store", "load", "save", default='store'}
       Action to perform. user can use ['write'|'store'] interchnageably. Both 
       does the same task. 
    as_frame: bool, default=False 
       Concert loaded array to data frame. `Columns` can be supplied 
       to construct the datafame. 
    columns: List, Optional 
       Columns used to construct the dataframe. When its given, it must be 
       consistent with the shape of the `arr` along axis 1 
       
    Returns 
    ---------
    None| data: ArrayLike or pd.DataFrame 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import array2hdf5
    >>> data = np.random.randn (100, 27 ) 
    >>> array2hdf5 ('test.h5', data   )
    >>> load_data = array2hdf5 ( 'test.h5', data, task ='load')
    >>> load_data.shape 
    Out[177]: (100, 27)
    """
    import_optional_dependency("h5py")
    import h5py 
    
    arr = is_iterable( arr, exclude_string =True, transform =True )
    act = copy.deepcopy(task)
    task = str(task).lower().strip() 
    
    if task in ("write", "store", "save"): 
        task ='store'
    assert task in {"store", "load"}, ("Expects ['store'|'load'] as task."
                                         f" Got {act!r}")
    # for consistency 
    arr = np.array ( arr )
    h5fname = str(filename).replace ('.h5', '')
    if task =='store': 
        if arr is None: 
            raise TypeError ("Array cannot be None when the task"
                             " consists to write a file.")
        with h5py.File(h5fname + '.h5', 'w') as hf:
            hf.create_dataset(dataname,  data=arr)
            
    elif task=='load': 
        with h5py.File(h5fname +".h5", 'r') as hf:
            data = hf[dataname][:]
            
        if  ellipsis2false( as_frame )[0]: 
            data = pd.DataFrame ( data , columns = columns )
            
    return data if task=='load' else None 

def remove_target_from_array(arr,/,  target_indices):
    """
    Remove specified columns from a 2D array based on target indices.

    This function extracts columns at specified indices from a 2D array, 
    returning the modified array without these columns and a separate array 
    containing the extracted columns. It raises an error if any of the indices
    are out of bounds.

    Parameters
    ----------
    arr : ndarray
        A 2D numpy array from which columns are to be removed.
    target_indices : list or ndarray
        Indices of the columns in `arr` that need to be extracted and removed.

    Returns
    -------
    modified_arr : ndarray
        The array obtained after removing the specified columns.
    target_arr : ndarray
        An array consisting of the columns extracted from `arr`.

    Raises
    ------
    ValueError
        If any of the target indices are out of the range of the array dimensions.

    Examples
    --------
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> target_indices = [1, 2]
    >>> modified_arr, target_arr = remove_target_from_array(arr, target_indices)
    >>> modified_arr
    array([[1],
           [4],
           [7]])
    >>> target_arr
    array([[2, 3],
           [5, 6],
           [7, 8]])
    """
    if any(idx >= arr.shape[1] for idx in target_indices):
        raise ValueError("One or more indices are out of the array's bounds.")

    target_arr = arr[:, target_indices]
    modified_arr = np.delete(arr, target_indices, axis=1)
    return modified_arr, target_arr

def extract_target(
    data: Union[ArrayLike, DataFrame],/, 
    target_names: Union[str, int, List[Union[str, int]]],
    drop: bool = True,
    columns: Optional[List[str]] = None,
) -> Tuple[Union[ArrayLike, Series, DataFrame], Union[ArrayLike, DataFrame]]:
    """
    Extracts specified target column(s) from a multidimensional numpy array
    or pandas DataFrame. 
    
    with options to rename columns in a DataFrame and control over whether the 
    extracted columns are dropped from the original data.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        The input data from which target columns are to be extracted. Can be a 
        NumPy array or a pandas DataFrame.
    target_names : Union[str, int, List[Union[str, int]]]
        The name(s) or integer index/indices of the column(s) to extract. 
        If `data` is a DataFrame, this can be a mix of column names and indices. 
        If `data` is a NumPy array, only integer indices are allowed.
    drop : bool, default True
        If True, the extracted columns are removed from the original `data`. 
        If False, the original `data` remains unchanged.
    columns : Optional[List[str]], default None
        If provided and `data` is a DataFrame, specifies new names for the 
        columns in `data`. The length of `columns` must match the number of 
        columns in `data`. This parameter is ignored if `data` is a NumPy array.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]
        A tuple containing two elements:
        - The extracted column(s) as a NumPy array or pandas Series/DataFrame.
        - The original data with the extracted columns optionally removed, as a
          NumPy array or pandas DataFrame.

    Raises
    ------
    ValueError
        If `columns` is provided and its length does not match the number of 
        columns in `data`.
        If any of the specified `target_names` do not exist in `data`.
        If `target_names` includes a mix of strings and integers for a NumPy 
        array input.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import extract_target
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> target, remaining = extract_target(df, 'B', drop=True)
    >>> print(target)
    0    4
    1    5
    2    6
    Name: B, dtype: int64
    >>> print(remaining)
       A  C
    0  1  7
    1  2  8
    2  3  9
    >>> arr = np.random.rand(5, 3)
    >>> target, modified_arr = extract_target(arr, 2, )
    >>> print(target)
    >>> print(modified_arr)
    """
    if isinstance (data, pd.Series): 
        data = data.to_frame() 
    if _is_arraylike_1d(data): 
        # convert to 2d array 
        data = data.reshape (-1, 1)
    
    is_frame = isinstance(data, pd.DataFrame)
    
    if is_frame and columns is not None:
        if len(columns) != data.shape[1]:
            raise ValueError("`columns` must match the number of columns in"
                             f" `data`. Expected {data.shape[1]}, got {len(columns)}.")
        data.columns = columns

    if isinstance(target_names, (int, str)):
        target_names = [target_names]

    if all(isinstance(name, int) for name in target_names):
        if max(target_names, default=-1) >= data.shape[1]:
            raise ValueError("All integer indices must be within the"
                             " column range of the data.")
    elif any(isinstance(name, int) for name in target_names) and is_frame:
        target_names = [data.columns[name] if isinstance(name, int) 
                        else name for name in target_names]

    if is_frame:
        missing_cols = [name for name in target_names 
                        if name not in data.columns]
        if missing_cols:
            raise ValueError(f"Column names {missing_cols} do not match "
                             "any column in the DataFrame.")
        target = data.loc[:, target_names]
        if drop:
            data = data.drop(columns=target_names)
    else:
        if any(isinstance(name, str) for name in target_names):
            raise ValueError("String names are not allowed for target names"
                             " when data is a NumPy array.")
        target = data[:, target_names]
        if drop:
            data = np.delete(data, target_names, axis=1)
            
    if  isinstance (target, np.ndarray): # squeeze the array 
        target = np.squeeze (target)
        
    return target, data

def _extract_target(
        X, target: Union[ArrayLike, int, str, List[Union[int, str]]]):
    """
    Extracts and validates the target variable(s) from the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset from which to extract the target variable(s).
    target : ArrayLike, int, str, or list of int/str
        The target variable(s) to be used. If an array-like or DataFrame, 
        it's directly used as `y`. If an int or str (or list of them), it 
        indicates the column(s) in `X` to be used as `y`.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        The dataset without the target column(s).
    y : pd.Series, np.ndarray, pd.DataFrame
        The target variable(s).
    target_names : list of str
        The names of the target variable(s) for labeling purposes.
    """
    target_names = []

    if isinstance(target, (list, pd.DataFrame)) or (
            isinstance(target, pd.Series) and not isinstance(X, np.ndarray)):
        if isinstance(target, list):  # List of column names or indexes
            if all(isinstance(t, str) for t in target):
                y = X[target]
                target_names = target
            elif all(isinstance(t, int) for t in target):
                y = X.iloc[:, target]
                target_names = [X.columns[i] for i in target]
            X = X.drop(columns=target_names)
        elif isinstance(target, pd.DataFrame):
            y = target
            target_names = target.columns.tolist()
            # Assuming target DataFrame is not part of X
        elif isinstance(target, pd.Series):
            y = target
            target_names = [target.name] if target.name else ["target"]
            if target.name and target.name in X.columns:
                X = X.drop(columns=target.name)
                
    elif isinstance(target, (int, str)):
        if isinstance(target, str):
            y = X.pop(target)
            target_names = [target]
        elif isinstance(target, int):
            y = X.iloc[:, target]
            target_names = [X.columns[target]]
            X = X.drop(columns=X.columns[target])
    elif isinstance(target, np.ndarray) or (
            isinstance(target, pd.Series) and isinstance(X, np.ndarray)):
        y = np.array(target)
        target_names = ["target"]
    else:
        raise ValueError("Unsupported target type or target does not match X dimensions.")
    
    check_consistent_length(X, y)
    
    return X, y, target_names

def categorize_target(
    arr : Union [ArrayLike , Series] , /, 
    func: _F = None,  
    labels: Union [int, List[int]] = None, 
    rename_labels: Optional[str] = None, 
    coerce:bool=False,
    order:str='strict',
    ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from gofast.tools.baseutils import categorize_target 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = categorize_target(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> categorize_target(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> categorize_target(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> categorize_target(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> categorize_target(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (
        arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`gofast.tools.baseutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to be categorized. If None or any other values, 
        the categorization of labels considers only the length of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly considers the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)    
   

def labels_validator(
    target: ArrayLike, 
    labels: Union[int, str, List[Union[int, str]]], 
    return_bool: bool = False
    ) -> Union[bool, List[Union[int, str]]]:
    """
    Validates if specified labels are present in the target array and 
    optionally returns a boolean indicating the presence of all labels or 
    the list of labels themselves.
    
    Parameters
    ----------
    target : np.ndarray
        The target array expected to contain the labels.
    labels : int, str, or list of int or str
        The label(s) supposed to be in the target array.
    return_bool : bool, default=False
        If True, returns a boolean indicating whether all specified 
        labels are present. If False, returns the list of labels.

    Returns
    -------
    bool or List[Union[int, str]]
        If `return_bool` is True, returns True if all labels are present, 
        False otherwise.
        If `return_bool` is False, returns the list of labels if all are present.
    
    Raises
    ------
    ValueError
        If any of the specified labels are missing in the target array and 
        `return_bool` is False.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import labels_validator
    >>> target = np.array([1, 2, 3, 4, 5])
    >>> labels_validator(target, [1, 2, 3])
    [1, 2, 3]
    >>> labels_validator(target, [0, 1], return_bool=True)
    False
    >>> labels_validator(target, 1)
    [1]
    >>> labels_validator(target, [6], return_bool=True)
    False
    """
    if isinstance(labels, (int, str)):
        labels = [labels]
    
    labels_present = np.unique([label for label in labels if label in target])
    missing_labels = [label for label in labels if label not in labels_present]

    if missing_labels:
        if return_bool:
            return False
        raise ValueError(f"Label{'s' if len(missing_labels) > 1 else ''}"
                        f" {', '.join(map(str, missing_labels))}"
                        f" {'are' if len(missing_labels) > 1 else 'is'}"
                        " missing in the target."
                    )

    return True if return_bool else labels

def generate_placeholders(
        iterable_obj: Iterable[_T]) -> List[str]:
    """
    Generates a list of string placeholders for each item in the input
    iterable. This can be useful for creating formatted string
    representations where each item's index is used within braces.

    :param iterable_obj: An iterable object (e.g., list, set, or any
        iterable collection) whose length determines the number of
        placeholders generated.
    :return: A list of strings, each representing a placeholder in
        the format "{n}", where n is the index of the placeholder.
        
    :Example:
        >>> from gofast.tools.baseutils import generate_placeholders
        >>> generate_placeholders_for_iterable({'ohmS', 'lwi', 'power', 'id', 
        ...                                     'sfi', 'magnitude'})
        ['{0}', '{1}', '{2}', '{3}', '{4}', '{5}']
    """
    return [f"{{{index}}}" for index in range(len(iterable_obj))]


def compute_set_operation( 
    iterable1: Iterable[Any],
    iterable2: Iterable[Any],
    operation: str = "intersection"
) -> Set[Any]:
    """
    Computes the intersection or difference between two iterable objects,
    returning the result as a set. This function is flexible and works
    with any iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object from which to compute the operation.
    iterable2 : Iterable[Any]
        The second iterable object.
    operation : str, optional
        The operation to perform, either 'intersection' or 'difference'.
        Defaults to 'intersection'.

    Returns
    -------
    Set[Any]
        A set of either common elements (intersection) or unique elements
        (difference) from the two iterables.

    Examples
    --------
    Intersection example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'}
    ... )
    {'b', 'c'}

    Difference example:
    >>> compute_set_operation(
    ...     ['a', 'b', 'c'], 
    ...     {'b', 'c', 'd'},
    ...     operation='difference'
    ... )
    {'a', 'd'}

    Notes
    -----
    The function supports only 'intersection' and 'difference' operations.
    It ensures the result is always returned as a set, regardless of the
    input iterable types.
    """
    
    set1 = set(iterable1)
    set2 = set(iterable2)

    if operation == "intersection":
        return set1 & set2  # Using & for intersection
    elif operation == "difference":
        # Returning symmetric difference
        return set1 ^ set2  # Using ^ for symmetric difference
    else:
        raise ValueError("Invalid operation specified. Choose either"
                         " 'intersection' or 'difference'.")

def find_intersection(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Set[Any]:
    """
    Computes the intersection of two iterable objects, returning a set
    of elements common to both. This function is designed to work with
    various iterable types, including lists, sets, and dictionaries.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Set[Any]
        A set of elements common to both `iterable1` and `iterable2`.

    Example
    -------
    >>> from gofast.tools.baseutils import find_intersection_between_generics
    >>> compute_intersection(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    {'ohmS', 'lwi', 'power'}

    Notes
    -----
    The result is always a set, regardless of the input types, ensuring
    that each element is unique and present in both iterables.
    """

    # Utilize set intersection operation (&) for clarity and conciseness
    return set(iterable1) & set(iterable2)

def find_unique_elements(
    iterable1: Iterable[Any],
    iterable2: Iterable[Any]
) -> Optional[Set[Any]]:
    """
    Computes the difference between two iterable objects, returning a set
    containing elements unique to the iterable with more unique elements.
    If both iterables contain an equal number of unique elements, the function
    returns None.

    This function is designed to work with various iterable types, including
    lists, sets, and dictionaries. The focus is on the count of unique elements
    rather than the total length, which allows for more consistent results
    across different types of iterables.

    Parameters
    ----------
    iterable1 : Iterable[Any]
        The first iterable object.
    iterable2 : Iterable[Any]
        The second iterable object.

    Returns
    -------
    Optional[Set[Any]]
        A set of elements unique to the iterable with more unique elements,
        or None if both have an equal number of unique elements.

    Example
    -------
    >>> find_unique_elements(
    ...     ['a', 'b', 'c', 'c'],
    ...     {'a', 'b'}
    ... )
    {'c'}

    Notes
    -----
    The comparison is based on the number of unique elements, not the
    iterable size. This approach ensures a more meaningful comparison
    when the iterables are of different types or when duplicates are present.
    """

    set1 = set(iterable1)
    set2 = set(iterable2)

    # Adjust the logic to focus on the uniqueness rather than size
    if len(set1) == len(set2):
        return None
    elif len(set1) > len(set2):
        return set1 - set2
    else:
        return set2 - set1

def validate_feature_existence(supervised_features: Iterable[_T], 
                               features: Iterable[_T]) -> None:
    """
    Validates the existence of supervised features within a list of all features.
    This is typically used to ensure that certain expected features (columns) are
    present in a pandas DataFrame.

    Parameters
    ----------
    supervised_features : Iterable[_T]
        An iterable of features presumed to be controlled or supervised.
        
    features : Iterable[_T]
        An iterable of all features, such as pd.DataFrame.columns.

    Raises
    ------
    ValueError
        If `supervised_features` are not found within `features`.
    """
    # Ensure input is in list format if strings are passed
    if isinstance(supervised_features, str):
        supervised_features = [supervised_features]
    if isinstance(features, str):
        features = [features]
    
    # Check for feature existence
    if not cfexist(features_to=supervised_features, features=list(features)):
        raise ValueError(f"Features {supervised_features} not found in {list(features)}")

def cfexist(features_to: List[Any], features: List[Any]) -> bool:
    """
    Checks if all elements of one list (features_to) exist within another list (features).

    Parameters
    ----------
    features_to : List[Any]
        List or array to be checked for existence within `features`.
        
    features : List[Any]
        List of whole features, e.g., as in pd.DataFrame.columns.

    Returns
    -------
    bool
        True if all elements in `features_to` exist in `features`, False otherwise.
    """
    # Normalize input to lists, handle string inputs
    if isinstance(features_to, str):
        features_to = [features_to]
    if isinstance(features, str):
        features = [features]

    # Check for existence
    return set(features_to).issubset(features)

def control_existing_estimator(
    estimator_name: str, 
    predefined_estimators=None, 
    raise_error: bool = False
) -> Union[Tuple[str, str], None]:
    """
    Validates and retrieves the corresponding prefix for a given estimator name.

    This function checks if the provided estimator name exists in a predefined
    list of estimators or in scikit-learn. If found, it returns the corresponding
    prefix and full name. Otherwise, it either raises an error or returns None,
    based on the 'raise_error' flag.

    Parameters
    ----------
    estimator_name : str
        The name of the estimator to check.
    predefined_estimators : dict, default _predefined_estimators
        A dictionary of predefined estimators.
    raise_error : bool, default False
        If True, raises an error when the estimator is not found. Otherwise, 
        emits a warning.

    Returns
    -------
    Tuple[str, str] or None
        A tuple containing the prefix and full name of the estimator, or 
        None if not found.

    Example
    -------
    >>> from gofast.tools.baseutils import control_existing_estimator
    >>> test_est = control_existing_estimator('svm')
    >>> print(test_est)
    ('svc', 'SupportVectorClassifier')
    """
    
    from ..exceptions import EstimatorError 
    # Define a dictionary of predefined estimators
    _predefined_estimators ={
            'dtc': ['DecisionTreeClassifier', 'dtc', 'dec', 'dt'],
            'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
            'sdg': ['SGDClassifier','sdg', 'sd', 'sdg'],
            'knn': ['KNeighborsClassifier','knn', 'kne', 'knr'],
            'rdf': ['RandomForestClassifier', 'rdf', 'rf', 'rfc',],
            'ada': ['AdaBoostClassifier','ada', 'adc', 'adboost'],
            'vtc': ['VotingClassifier','vtc', 'vot', 'voting'],
            'bag': ['BaggingClassifier', 'bag', 'bag', 'bagg'],
            'stc': ['StackingClassifier','stc', 'sta', 'stack'],
            'xgb': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm', 'xgb'], 
          'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
          'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
            }
    predefined_estimators = predefined_estimators or _predefined_estimators
    
    estimator_name= estimator_name.lower().strip() if isinstance (
        estimator_name, str) else get_estimator_name(estimator_name)
    
    # Check if the estimator is in the predefined list
    for prefix, names in predefined_estimators.items():
        lower_names = [name.lower() for name in names]
        
        if estimator_name in lower_names:
            return prefix, names[0]

    # If not found in predefined list, check if it's a valid scikit-learn estimator
    if estimator_name in _get_sklearn_estimator_names():
        return estimator_name, estimator_name

    # If XGBoost is installed, check if it's an XGBoost estimator
    if 'xgb' in predefined_estimators and estimator_name.startswith('xgb'):
        return 'xgb', estimator_name

    # If raise_error is True, raise an error; otherwise, emit a warning
    if raise_error:
        valid_names = [name for names in predefined_estimators.values() for name in names]
        raise EstimatorError(f'Unsupported estimator {estimator_name!r}. '
                             f'Expected one of {valid_names}.')
    else:
        available_estimators = _get_available_estimators(predefined_estimators)
        warning_msg = (f"Estimator {estimator_name!r} not found. "
                       f"Expected one of: {available_estimators}.")
        warnings.warn(warning_msg)

    return None

def _get_sklearn_estimator_names():
    
    # Retrieve all scikit-learn estimator names using all_estimators
    sklearn_estimators = [name for name, _ in all_estimators(type_filter='classifier')]
    sklearn_estimators += [name for name, _ in all_estimators(type_filter='regressor')]
    return sklearn_estimators

def _get_available_estimators(predefined_estimators):
    # Combine scikit-learn and predefined estimators
    sklearn_estimators = _get_sklearn_estimator_names()
    xgboost_estimators = ['xgb' + name for name in predefined_estimators['xgb']]
    
    available_estimators = sklearn_estimators + xgboost_estimators
    return available_estimators

def get_target(df, tname, inplace=True):
    """
    Extracts one or more target columns from a DataFrame and optionally
    modifies the original DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to extract the target(s).
    tname : str or list of str
        The name(s) of the target column(s) to extract. These must be present
        in the DataFrame.
    inplace : bool, optional
        If True, the DataFrame is modified in place by removing the target
        column(s). Defaults to True.

    Returns
    -------
    tuple
        A tuple containing:
        - pd.Series or pd.DataFrame: The extracted target column(s).
        - pd.DataFrame: The modified or unmodified DataFrame depending on the
          `inplace` parameter.

    Raises
    ------
    ValueError
        If any of the specified target names are not in the DataFrame columns.
    TypeError
        If `df` is not a pandas DataFrame.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.baseutils import get_target 
    >>> data = load_iris(as_frame=True).frame
    >>> targets, modified_df = get_target(data, 'target', inplace=False)
    >>> print(targets.head())
    >>> print(modified_df.columns)

    Notes
    -----
    This function is particularly useful when preparing data for machine
    learning models, where separating features from labels is a common task.

    See Also
    --------
    extract_target : Similar function with enhanced capabilities for handling
                     more complex scenarios.
    
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if isinstance(tname, str):
        tname = [tname]  # Convert string to list for uniform processing

    missing_columns = [name for name in tname if name not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Target name(s) not found in DataFrame columns: {missing_columns}")

    target_data = df[tname]
    if inplace:
        df.drop(tname, axis=1, inplace=True)

    return target_data, df

@Dataify(auto_columns=True )
def binning_statistic(
    data, categorical_column, 
    value_column, 
    statistic='mean'
    ):
    """
    Compute a statistic for each category in a categorical column of a dataset.

    This function categorizes the data into bins based on a categorical variable and then
    applies a statistical function to the values of another column for each category.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    value_column : str
        Name of the column in `data` from which the statistic will be calculated.
    statistic : str, optional
        The statistic to compute (default is 'mean'). Other options include 
        'sum', 'count','median', 'min', 'max', etc.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import binning_statistic
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Value': [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> binning_statistic(df, 'Category', 'Value', statistic='mean')
       Category  Mean_Value
    0        A         3.33
    1        B         3.50
    2        C         5.50
    """
    if statistic not in ('mean', 'sum', 'count', 'median', 'min',
                         'max', 'proportion'):
        raise ValueError(
            "Unsupported statistic. Please choose from 'mean',"
            " 'sum', 'count', 'median', 'min', 'max', 'proportion'.")

    is_categorical(data, categorical_column)
    exist_features(data, features =value_column, name ="value_column")
    grouped_data = data.groupby(categorical_column)[value_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{value_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{value_column}')
    elif statistic == 'count':
        result = grouped_data.count().reset_index(name=f'Count_{value_column}')
    elif statistic == 'median':
        result = grouped_data.median().reset_index(name=f'Median_{value_column}')
    elif statistic == 'min':
        result = grouped_data.min().reset_index(name=f'Min_{value_column}')
    elif statistic == 'max':
        result = grouped_data.max().reset_index(name=f'Max_{value_column}')
    elif statistic == 'proportion':
        total_count = data[value_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{value_column}')
        
    return result

@Dataify(auto_columns=True)
def category_count(data, /, *categorical_columns, error='raise'):
    """
    Count occurrences of each category in one or more categorical columns 
    of a dataset.
    
    This function computes the frequency of each unique category in the specified
    categorical columns of a pandas DataFrame and handles different ways of error
    reporting including raising an error, warning, or ignoring the error when a
    specified column is not found.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    *categorical_columns : str
        One or multiple names of the columns in `data` which contain the 
        categorical variables.
    error : str, optional
        Error handling strategy - 'raise' (default), 'warn', or 'ignore' which
        dictates the action when a categorical column is not found.

    Returns
    -------
    counts : DataFrame
        A DataFrame with each category and the corresponding count from each
        categorical column. If multiple columns are provided, columns are named as
        'Category_i' and 'Count_i'.

    Raises
    ------
    ValueError
        If any categorical column is not found in the DataFrame and error is 'raise'.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import category_count
    >>> df = pd.DataFrame({
    ...     'Fruit': ['Apple', 'Banana', 'Apple', 'Cherry', 'Banana', 'Apple'],
    ...     'Color': ['Red', 'Yellow', 'Green', 'Red', 'Yellow', 'Green']
    ... })
    >>> category_count(df, 'Fruit', 'Color')
       Category_1  Count_1 Category_2  Count_2
    0      Apple        3        Red        2
    1     Banana        2     Yellow        2
    2     Cherry        1      Green        2
    >>> category_count(df, 'NonExistentColumn', error='warn')
    Warning: Column 'NonExistentColumn' not found in the dataframe.
    Empty DataFrame
    Columns: []
    Index: []
    """
    results = []
    for i, column in enumerate(categorical_columns, 1):
        if column not in data.columns:
            message = f"Column '{column}' not found in the dataframe."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                continue
            elif error == 'ignore':
                continue

        count = data[column].value_counts().reset_index()
        count.columns = [f'Category_{i}', f'Count_{i}']
        results.append(count)

    if not results:
        return pd.DataFrame()

    # Merge all results into a single DataFrame
    final_df = functools.reduce(lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how='outer'), results)
    final_df.fillna(value=np.nan, inplace=True)
    
    if len( results)==1: 
        final_df.columns =['Category', 'Count']
    return final_df

@Dataify(auto_columns=True) 
def soft_bin_stat(
    data, /, categorical_column, 
    target_column, 
    statistic='mean', 
    update=False, 
    ):
    """
    Compute a statistic for each category in a categorical 
    column based on a binary target.

    This function calculates statistics like mean, sum, or proportion 
    for a binary target variable, grouped by categories in a 
    specified column.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    target_column : str
        Name of the column in `data` which contains the binary target variable.
    statistic : str, optional
        The statistic to compute for the binary target (default is 'mean').
        Other options include 'sum' and 'proportion'.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding 
        computed statistic.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import soft_bin_stat
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Target': [1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> soft_bin_stat(df, 'Category', 'Target', statistic='mean')
       Category  Mean_Target
    0        A     0.666667
    1        B     0.500000
    2        C     0.500000

    >>> soft_bin_stat(df.values, 'col_0', 'col_1', statistic='mean')
      col_0  Mean_col_1
    0     A    0.666667
    1     B    0.500000
    2     C    0.500000
    """
    if statistic not in ['mean', 'sum', 'proportion']:
        raise ValueError("Unsupported statistic. Please choose from "
                         "'mean', 'sum', 'proportion'.")
    
    is_categorical(data, categorical_column)
    exist_features(data, features= target_column, name ='Target')
    grouped_data = data.groupby(categorical_column)[target_column]
    
    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{target_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{target_column}')
    elif statistic == 'proportion':
        total_count = data[target_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{target_column}')

    return result

def reshape_to_dataframe(flattened_array, columns, error ='raise'):
    """
    Reshapes a flattened array into a pandas DataFrame or Series based on the
    provided column names. If the number of columns does not allow reshaping
    to match the array length, it raises an error.

    Parameters
    ----------
    flattened_array : array-like
        The flattened array to reshape.
    columns : list of str
        The list of column names for the DataFrame. If a single name is provided,
        a Series is returned.
        
    error : {'raise', 'warn', 'ignore'}, default 'raise'
        Specifies how to handle the situation when the number of elements in the
        flattened array is not compatible with the number of columns required for
        reshaping. Options are:
        
        - 'raise': Raises a ValueError. This is the default behavior.
        - 'warn': Emits a warning, but still returns the original flattened array.
        - 'ignore': Does nothing about the error, just returns the original
          flattened array.
        
    Returns
    -------
    pandas.DataFrame or pandas.Series
        A DataFrame or Series reshaped according to the specified columns.

    Raises
    ------
    ValueError
        If the total number of elements in the flattened array does not match
        the required number for a complete reshaping.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.baseutils import reshape_to_dataframe
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> print(reshape_to_dataframe(data, ['A', 'B', 'C']))  # DataFrame with 2 rows and 3 columns
    >>> print(reshape_to_dataframe(data, 'A'))  # Series with 6 elements
    >>> print(reshape_to_dataframe(data, ['A']))  # DataFrame with 6 rows and 1 column
    """
    # Check if the reshaping is possible
    is_string = isinstance ( columns, str )
    # Convert single string column name to list
    if isinstance(columns, str):
        columns = [columns]
        
    num_elements = len(flattened_array)
    num_columns = len(columns)
    if num_elements % num_columns != 0:
        message = ("The number of elements in the flattened array is not"
                   " compatible with the number of columns.")
        if error =="raise": 
            raise ValueError(message)
        elif error =='warn': 
            warnings.warn(message, UserWarning)
        return flattened_array
    # Calculate the number of rows that will be needed
    num_rows = num_elements // num_columns

    # Reshape the array
    reshaped_array = np.reshape(flattened_array, (num_rows, num_columns))

    # Check if we need to return a DataFrame or a Series
    if num_columns == 1 and is_string:
        return pd.Series(reshaped_array[:, 0], name=columns[0])
    else:
        return pd.DataFrame(reshaped_array, columns=columns)

def save_figure(fig, filename=None, dpi=300, close=True, ax=None, 
                tight_layout=False, bbox_inches='tight'):
    """
    Saves a matplotlib figure to a file and optionally closes it. 
    Automatically generates a unique filename if not provided.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str, optional
        The name of the file to save the figure to. If None, a unique name
        is generated based on the current date-time.
    dpi : int, optional
        The resolution of the output file in dots per inch.
    close : bool, optional
        Whether to close the figure after saving.
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        Axes object(s) to perform operations on before saving.
    tight_layout : bool, optional
        Whether to adjust subplot parameters to give specified padding.
    bbox_inches : str, optional
        Bounding box in inches: 'tight' or a specific value.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from gofast.tools.baseutils import save_figure
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> ax.plot(x, y)
    >>> save_figure(fig, close=True, ax=ax)

    Notes
    -----
    If the filename is not specified, this function generates a filename that
    is unique to the second, using the pattern 'figure_YYYYMMDD_HHMMSS.png'.
    If two figures are saved within the same second, it appends microseconds
    to ensure uniqueness.
    """
    # Generate a unique filename if not provided
    if filename is None:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_{date_time}.png"
        while os.path.exists(filename):
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"figure_{date_time}.png"

    # Adjust layout if requested
    if tight_layout:
        fig.tight_layout()

    # Optionally adjust axis properties
    if ax is not None:
        if isinstance(ax, (list, tuple, np.ndarray)):
            for a in ax:
                a.grid(True)
        else:
            ax.grid(True)
    
    # Save the figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved as '{filename}' with dpi={dpi}")

    # Optionally close the figure
    if close:
        plt.close(fig)
        print("Figure closed.")
