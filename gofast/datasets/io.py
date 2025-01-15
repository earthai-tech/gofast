# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio
"""
Base IO functions for managing all the datasets.
Provides functions for loading, retrieving, and managing dataset files, 
including CSV and text files, as well as dataset descriptions.
"""

import os 
import csv 
import shutil 
import warnings 
from pathlib import Path 
from importlib import resources
from collections import namedtuple
from typing import Union, List, Optional, Tuple, Any
from urllib.parse import urljoin
from gofast.utils.base_utils import check_file_exists, fancier_downloader

import numpy as np 
import pandas as pd 

from ..api.structures import Boxspace  
from ..core.checks import random_state_validator, is_iterable, exist_features
from ..utils.base_utils import is_readable 

__all__=[
    'csv_data_loader',
    'description_loader',
    'get_data',
    'remove_data',
    'text_files_loader'
 ]

DMODULE = "gofast.datasets.data" ; DESCR = "gofast.datasets.descr"

# create a namedtuple for remote data and url 
RemoteMetadata = namedtuple("RemoteMetadata", ["file", "url", "checksum", "descr", "module"])
RemoteDataURL ='https://raw.githubusercontent.com/earthai-tech/gofast/master/gofast/datasets/data/'

RemoteMetadata.descr =DESCR 
RemoteMetadata.url= RemoteDataURL 
RemoteMetadata.module = DMODULE 

def get_data(data =None) -> str: 
    if data is None:
        data = os.environ.get("GOFAST_DATA", os.path.join("~", "gofast_data"))
    data = os.path.expanduser(data)
    os.makedirs(data, exist_ok=True)
    return data

get_data.__doc__ ="""\
Get the data from home directory  and return gofast data directory 

By default the data directory is set to a folder named 'gofast_data' in the
user home folder. Alternatively, it can be set by the 'gofast_DATA' environment
variable or programmatically by giving an explicit folder path. The '~'
symbol is expanded to the user home folder.
If the folder does not already exist, it is automatically created.

Parameters
----------
data : str, default=None
    The path to gofast data directory. If `None`, the default path
    is `~/gofast_data`.
Returns
-------
data: str
    The path to gofast data directory.

"""
def remove_data(data=None): #clear 
    """Delete all the content of the data home cache.
    
    Parameters
    ----------
    data : str, default=None
        The path to gofast data directory. If `None`, the default path
        is `~/gofast_data`.
    """
    data = get_data(data)
    shutil.rmtree(data)

def download_file_if(
    metadata: Optional[RemoteMetadata],
    package_name: str = DMODULE,
    error: str = 'raise',
    verbose: bool = True
) -> None:
    """
    Download a file from a remote URL if it does not exist in the specified package.

    Parameters
    ----------
    metadata : Optional[RemoteMetadata]
        Metadata of the remote file. If a string is provided instead of a 
        `RemoteMetadata` instance, it is treated as the filename, and other 
        attributes are set to default values.
    
    package_name : str, default=DMODULE
        The name of the package where the file should be stored.
    
    error : str, default='raise'
        Determines the behavior when a download fails. 
        - `'warn'`: Emit a warning and continue.
        - `'raise'`: Raise an exception.
    
    verbose : bool, default=True
        If `True`, prints messages about the download status.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the `error` parameter is not set to `'warn'` or `'raise'`.
    RuntimeError
        If the download fails and `error` is set to `'raise'`.
    """
    # Validate the 'error' parameter
    if error not in ['warn', 'raise', 'ignore']:
        raise ValueError("`error` parameter must be either 'warn' or 'raise'.")

    # If metadata is a string, convert it to a RemoteMetadata instance with default values
    if isinstance(metadata, str):
        metadata = RemoteMetadata(
            file=metadata,            # Set 'file' to the provided string
            url=RemoteDataURL,        # Use the default remote data URL
            checksum='',              # Default checksum (can be updated as needed)
            descr=DESCR,              # Default description
            module=DMODULE            # Default module path
        )
    
    # Ensure metadata is now a RemoteMetadata instance
    if not isinstance(metadata, RemoteMetadata):
        raise TypeError("`metadata` must be a string or a RemoteMetadata instance.")
    
    # Check if the specified file already exists within the given package
    file_exists = check_file_exists(package_name, metadata.file)
    
    if not file_exists:
        try:
            # Construct the full path to the file within the package using importlib.resources
            package_path = str(resources.files(package_name).joinpath(metadata.file))
            
            # Determine the directory where the file should be saved
            destination_dir = os.path.dirname(package_path)
            
            # Ensure the destination directory exists to prevent errors during download
            os.makedirs(destination_dir, exist_ok=True)
            
            # Construct the full URL to download the file by joining the base URL and filename
            file_url = urljoin(metadata.url, metadata.file)
            
            # Download the file using the fancier_downloader utility
            fancier_downloader(
                url=file_url,               # URL from which to download the file
                filename=metadata.file,     # Name of the file to save
                dstpath=destination_dir,     # Destination directory path
                check_size=True, 
                verbose=False, 
            )
            
            # Inform the user about the successful download if verbose is True
            if verbose:
                print(
                    f"\nSuccessfully downloaded '{metadata.file}' "
                    f"to '{destination_dir}'."
                )
        except Exception as e:
            # Handle errors based on the 'error' parameter
            if error == 'warn':
                # Emit a warning without stopping the program
                warnings.warn(
                    f"\nFailed to download '{metadata.file}'. Error: {e}"
                )
            elif error == 'raise':
                # Raise a RuntimeError with the original exception as context
                raise RuntimeError(
                    f"Failed to download '{metadata.file}'. Check your "
                    "connection and retry."
                ) from e
    else:
        # Inform the user that the file already exists and skip the download
        if verbose:
            warnings.warn(
                f"File '{metadata.file}' already exists in package '{package_name}'. "
                "Skipping download."
            )

def to_dataframe(data, target_names=None, feature_names=None, target=None):
    """
    Refines input data into a structured DataFrame, distinguishing between
    features and targets based on specified column names or a separate target array.

    Parameters
    ----------
    data : str, path-like object, DataFrame, or array-like
        Source data, which can be a file path, a DataFrame, or an array-like object.
    target_names : list of str, optional
        Column names in `data` designated as targets. If specified, these columns
        are separated into the target DataFrame `y`.
    feature_names : list of str, optional
        Column names to retain as features in the output DataFrame `X`. If not provided,
        all columns except those specified in `tnames` are used.
    target : ndarray, pd.Series, pd.DataFrame, or None, optional
        Explicit target data. If provided, this will be used as the target DataFrame `y`,
        potentially in addition to any targets specified in `target_names`.

    Returns
    -------
    DataFrame
        The combined DataFrame including both features and target(s),
        if applicable.
    DataFrame
        The features DataFrame `X`.
    DataFrame or None
        The target DataFrame `y`, if targets are specified either through
        `tnames` or `target`.

    Examples
    --------
    >>> from gofast.dataset.io import _to_dataframe
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6],
    ...     'target': [0, 1, 0]
    ... })
    >>> feature_names = ['feature1', 'feature2']
    >>> tnames = ['target']
    >>> combined, X, y = _to_dataframe(data, tnames, feature_names)
    >>> X
       feature1  feature2
    0         1         4
    1         2         5
    2         3         6
    >>> y
       target
    0      0
    1      1
    2      0
    """
    print(data.columns )
    if isinstance(data, (str, bytes)):
        data = is_readable(data)  # Assumes CSV; adjust as necessary.
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        try:
            data = pd.DataFrame(data, columns=feature_names)
        except: data =pd.DataFrame(data )
        
    if target_names is not None: 
        target_names = is_iterable (
            target_names, exclude_string=True, transform=True)
    print(data.columns )
    feature_data = data.drop(
        columns=target_names, errors='ignore') if target_names else data
    try:
        exist_features(data, target_names) # check whether tnames is on data 
    except:
        target_from_data= pd.DataFrame()
    else: 
        target_from_data = data[target_names] if target_names else pd.DataFrame()
    
    if target is not None:
        if isinstance(target, (pd.Series, pd.DataFrame, np.ndarray)):
            if isinstance(target, np.ndarray):
                target = pd.DataFrame(
                    target, columns=target_names
                    ) if target_names else pd.DataFrame(target)
   
            elif isinstance(target, pd.Series):
                target = pd.DataFrame(target)
            # No need to filter out duplicates here, as we've preemptively handled them.
            # Concatenate target_from_data and new target, handling duplicate columns
            # target = pd.concat([target_from_data, target], axis=1
            #                     ).loc[:, ~target.columns.duplicated()]
            
            # Before concatenating, ensure that the columns from 
            #  'target_from_data' and 'target' do not overlap.
            # This avoids the boolean indexing issue by never creating 
            # duplicate columns in the first place.
            # First, identify the duplicate columns that would result 
            # from concatenation.
            if not target_from_data.empty:
                duplicate_columns = target_from_data.columns.intersection(
                    target.columns)
                # Drop duplicate columns from one of the DataFrames
                # before concatenation.
                target_from_data = target_from_data.drop(
                    columns=duplicate_columns, errors='ignore')
            # Now, concatenate 'target_from_data' and 'target' without 
            # risking duplicate columns.
            target = pd.concat([target_from_data, target], axis=1)
        else:
            raise TypeError("The 'target' parameter must be an ndarray,"
                            " pd.Series, pd.DataFrame, or None.")
    else:
        target = target_from_data

    # for consistency, recheck the feature data. 
    feature_data =( 
        feature_data [feature_names] 
        if feature_names is not None else feature_data 
        )
    # Ensure that `target` is either a DataFrame or None for 
    # consistent return types
    target = target if not target.empty else None
    combined = pd.concat(
        [feature_data, target], axis=1) if target is not None else feature_data

    return combined, feature_data, target

def _to_dataframe(
    data: Union[str, Path, pd.DataFrame, np.ndarray, List[Any]],
    target_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    target: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None, 
    error: str = 'warn'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Refines input data into structured DataFrames for features and targets.

    This function processes input data, distinguishing between features and targets
    based on specified column names or a separate target array. It ensures that
    the resulting DataFrames are well-structured, free of duplicate columns,
    and align with the provided feature and target specifications.

    Parameters
    ----------
    data : str, Path, pd.DataFrame, np.ndarray, list
        Source data, which can be:
        - A file path (string or Path object) pointing to a CSV file.
        - A pandas DataFrame.
        - An array-like object (e.g., list or NumPy array).
    target_names : list of str, optional
        Column names in `data` designated as targets. If specified, these columns
        are separated into the target DataFrame `y`.
    feature_names : list of str, optional
        Column names to retain as features in the output DataFrame `X`. If not provided,
        all columns except those specified in `target_names` are used.
    target : np.ndarray, pd.Series, pd.DataFrame, optional
        Explicit target data. If provided, this will be used as the target DataFrame `y`,
        potentially in addition to any targets specified in `target_names`.

    error: {'raise', 'warn', 'ignore'}, default ='warn' 
        Error policy for managing the data and target transformations. 
        - if ``error=='raise'``, raise (ValueError or TypeError) if target, the 
          target columns or the data does not meet any conditions of their 
          structures. 
        - if ``error='warn', find solution to handle it and continue processing 
        - if ``error='ignore'`` continue runing silently. 
        
    Returns
    -------
    combined : pd.DataFrame
        The combined DataFrame including both features and target(s), if applicable.
    X : pd.DataFrame
        The features DataFrame.
    y : pd.DataFrame or None
        The target DataFrame, if targets are specified either through
        `target_names` or `target`. Returns `None` if no target is specified.

    Raises
    ------
    ValueError
        If input data cannot be interpreted as a DataFrame.
        If specified `target_names` or `feature_names` are not present in the data.
        If `target` is provided but its dimensions or columns are incompatible.
    TypeError
        If `target` is not an instance of `np.ndarray`, `pd.Series`,
        `pd.DataFrame`, or `None`.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6],
    ...     'target': [0, 1, 0]
    ... })
    >>> feature_names = ['feature1', 'feature2']
    >>> target_names = ['target']
    >>> combined, X, y = to_dataframe(data, target_names, feature_names)
    >>> X
       feature1  feature2
    0         1         4
    1         2         5
    2         3         6
    >>> y
       target
    0      0
    1      1
    2      0

    >>> # Using separate target
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3],
    ...     'feature2': [4, 5, 6]
    ... })
    >>> target = pd.Series([0, 1, 0], name='target')
    >>> combined, X, y = to_dataframe(data, target=target)
    >>> X
       feature1  feature2
    0         1         4
    1         2         5
    2         3         6
    >>> y
       target
    0      0
    1      1
    2      0

    >>> # Loading from CSV file
    >>> combined, X, y = to_dataframe(
    ...    'data.csv', target_names=['target'],
    ...     feature_names=['feature1', 'feature2']
    ...  )
    """

    # Step 1: Load data into DataFrame
    if isinstance(data, (str, Path, bytes)):
        path = Path(data)
        data_df = is_readable(path)
    elif isinstance(data, pd.DataFrame):
        data_df = data.copy()
    elif isinstance(data, (np.ndarray, list)):
        if feature_names is not None:
            if isinstance(data, list) and not all(
                isinstance(row, (list, tuple, np.ndarray))
                for row in data
            ):
                # Reshape if list of scalars
                data = np.array(data).reshape(-1, 1)
            elif isinstance(data, np.ndarray) and data.ndim == 1:
                data = data.reshape(-1, 1)
            if len(feature_names) != data.shape[1]:
                raise ValueError(
                    f"Length of 'feature_names' ({len(feature_names)}) "
                    f"does not match number of columns in 'data' "
                    f"({data.shape[1]})."
                )
            try:
                data_df = pd.DataFrame(data, columns=feature_names)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert array-like data to DataFrame "
                    f"with provided 'feature_names': {e}"
                )
        else:
            try:
                data_df = pd.DataFrame(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert array-like data to DataFrame: {e}"
                )
    else:
        raise TypeError(
            "The 'data' parameter must be a file path (str or Path), "
            "a pandas DataFrame, or an array-like object."
        )

    # Step 2: Separate target columns if target_names are provided
    if target_names is not None:
        target_names = is_iterable(
            target_names, exclude_string=True, transform=True
        )
        missing_targets = [
            name for name in target_names if name not in data_df.columns
        ]
        if missing_targets:
            err_msg = (
                f"The following target columns are not in data: "
                f"{missing_targets}"
            )
            if error == 'raise':
                raise ValueError(err_msg)
            elif error == 'warn':
                warnings.warn(err_msg)
        # Keep only valid target_names
        valid_target_names = [
            name for name in target_names if name in data_df.columns
        ]
        y_from_data = data_df[valid_target_names].copy()
        X = data_df.drop(columns=valid_target_names)
    else:
        y_from_data = pd.DataFrame()
        X = data_df.copy()

    # Step 3: Select feature columns if feature_names are provided
    if feature_names is not None:
        feature_names = is_iterable(
            feature_names, exclude_string=True, transform=True
        )
        missing_features = [
            name for name in feature_names if name not in X.columns
        ]
        if missing_features:
            err_msg = (
                f"The following feature columns are not in data: "
                f"{missing_features}"
            )
            if error == 'raise':
                raise ValueError(err_msg)
            elif error == 'warn':
                warnings.warn(err_msg)
        # Keep only valid feature_names
        valid_feature_names = [
            name for name in feature_names if name in X.columns
        ]
        X = X[valid_feature_names].copy()

    # Step 4: Handle separate target if provided
    if target is not None:
        if not isinstance(
            target, (np.ndarray, pd.Series, pd.DataFrame)
        ):
            if error == 'raise':
                raise TypeError(
                    "The 'target' must be an ndarray, pd.Series, "
                    "pd.DataFrame, or None."
                )
            elif error == 'warn':
                warnings.warn(
                    f"The 'target' {type(target).__name__!r} detected. "
                    "Should be converted to numpy array."
                )
                target = np.array(target)
        
        if isinstance(target, np.ndarray):
            if target_names is not None:
                if target.ndim == 1 and len(target_names) == 1:
                    target_df = pd.DataFrame(
                        target, columns=target_names
                    )
                elif target.ndim == 2 and target.shape[1] == len(target_names):
                    target_df = pd.DataFrame(
                        target, columns=target_names
                    )
                else:
                    if error == 'raise':
                        raise ValueError(
                            "Shape of 'target' does not match 'target_names'."
                        )
                    elif error == 'warn':
                        warnings.warn(
                            "Shape of 'target' does not match 'target_names'."
                        )
                        target_df = pd.DataFrame(target)
            else:
                target_df = pd.DataFrame(target)
        elif isinstance(target, pd.Series):
            target_df = target.to_frame()
            if target_names is not None:
                if len(target_names) != 1:
                    raise ValueError(
                        "Length of 'target_names' must be 1 when "
                        "target is a pd.Series."
                    )
                target_df.columns = target_names
        elif isinstance(target, pd.DataFrame):
            if target_names is not None:
                missing_target_cols = [
                    name for name in target_names if name not in target.columns
                ]
                if missing_target_cols:
                    err_msg = (
                        f"The following target columns are missing in 'target' "
                        f"DataFrame: {missing_target_cols}"
                    )
                    if error == 'raise':
                        raise ValueError(err_msg)
                    elif error == 'warn':
                        warnings.warn(err_msg)
                # Keep only valid target_names
                valid_target_cols = [
                    name for name in target_names if name in target.columns
                ]
                target_df = target[valid_target_cols].copy()
            else:
                target_df = target.copy()
        
        # Combine y_from_data and target_df
        if not y_from_data.empty:
            overlapping = y_from_data.columns.intersection(target_df.columns)
            if not overlapping.empty:
                y_from_data = y_from_data.drop(
                    columns=overlapping
                )
            y_combined = pd.concat(
                [y_from_data, target_df], axis=1
            )
        else:
            y_combined = target_df.copy()
    else:
        y_combined = y_from_data.copy()

    # Step 5: Final checks and return
    if not y_combined.empty:
        y_final = y_combined.copy()
    else:
        y_final = None

    # Combine X and y_final
    if y_final is not None:
        combined = pd.concat([X, y_final], axis=1)
    else:
        combined = X.copy()

    return combined, X, y_final


def csv_data_loader(
    data_file,*, data_module=DMODULE, descr_file=None, descr_module=DESCR,
    include_headline =False, 
):
    feature_names= None # expect to extract features as the head columns 
    cat_feature_exist = False # for homogeneous feature control
    with resources.files(data_module).joinpath(
            data_file).open('r', encoding='utf-8') as csv_file: #python >3.8
    # with resources.open_text(data_module, data_file, 
    #                          encoding='utf-8') as csv_file: # Python 3.8
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0]) ; n_features = int(temp[1])
        # remove empty if exist in the list 
        tnames = np.array(list(filter(None,temp[2:])))
        # to prevent an error change the datatype to string 
        data = np.empty((n_samples, n_features)).astype('<U99')
        target = np.empty((n_samples,), dtype=float)
        if include_headline: 
            # remove target  expected to be located to the last columns
            feature_names = list(next(data_file)[:-1])  
            
        #XXX TODO: move the target[i] to try exception if target is str 
        for i, ir in enumerate(data_file):
            try : 
                data[i] = np.asarray(ir[:-1], dtype=float)
            except ValueError: 
                data[i] = np.asarray(ir[:-1], dtype ='<U99' ) # dont convert anything 
                cat_feature_exist = True # mean cat feature exists
            target[i] = np.asarray(ir[-1], dtype=float)
            
    if not cat_feature_exist: 
        # reconvert the datatype to float 
        data = data.astype (float)
    # reconvert target if problem is classification rather than regression 
    try : 
        target =target.astype(int )
    except : pass  
    
    if descr_file is None:
        return data, target, tnames, feature_names
    else:
        assert descr_module is not None
        descr = description_loader(descr_module=descr_module, descr_file=descr_file)
        return data, target, tnames, feature_names,  descr

csv_data_loader.__doc__="""\
Loads `data_file` from `data_module with `importlib.resources`.

Parameters
----------
data_file: str
    Name of csv file to be loaded from `data_module/data_file`.
    For example `'bagoue.csv'`.
data_module : str or module, default='gofast.datasets.data'
    Module where data lives. The default is `'gofast.datasets.data'`.
descr_file_name : str, default=None
    Name of rst file to be loaded from `descr_module/descr_file`.
    For example `'bagoue.rst'`. See also :func:`description_loader`.
    If not None, also returns the corresponding description of
    the dataset.
descr_module : str or module, default='gofast.datasets.descr'
    Module where `descr_file` lives. See also :func:`description_loader`.
    The default is `'gofast.datasets.descr'`.
Returns
-------
data : ndarray of shape (n_samples, n_features)
    A 2D array with each row representing one sample and each column
    representing the features of a given sample.
target : ndarry of shape (n_samples,)
    A 1D array holding target variables for all the samples in `data`.
    For example target[0] is the target variable for data[0].
target_names : ndarry of shape (n_samples,)
    A 1D array containing the names of the classifications. For example
    target_names[0] is the name of the target[0] class.
descr : str, optional
    Description of the dataset (the content of `descr_file_name`).
    Only returned if `descr_file` is not None.

"""

def description_loader(descr_file, *, descr_module=DESCR, encoding ='utf8'):
    # fdescr=resources.files(descr_module).joinpath(descr_file).read_text(
    #     encoding=encoding)
    fdescr = resources.read_text(descr_module, descr_file, encoding= 'utf8')
    return fdescr

description_loader.__doc__ ="""\
Load `descr_file` from `descr_module` with `importlib.resources`.
 
Parameters
----------
descr_file_name : str, default=None
    Name of rst file to be loaded from `descr_module/descr_file`.
    For example `'bagoue.rst'`. See also :func:`description_loader`.
    If not None, also returns the corresponding description of
    the dataset.
descr_module : str or module, default='gofast.datasets.descr'
    Module where `descr_file` lives. See also :func:`description_loader`.
    The default  is `'gofast.datasets.descr'`.
     
Returns
-------
fdescr : str
    Content of `descr_file_name`.

"""

def text_files_loader(
    container_path,*,description=None,categories=None,
    load_content=True,shuffle=True,encoding=None,decode_error="strict",
    random_state=42, allowed_extensions=None,
):
    target = []
    target_names = []
    filenames = []

    folders = [
        f for f in sorted(os.listdir(container_path)) if os.path.isdir(
            os.path.join(container_path, f))
    ]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    if allowed_extensions is not None:
        allowed_extensions = frozenset(allowed_extensions)

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = os.path.join(container_path, folder)
        files = sorted(os.listdir(folder_path))
        if allowed_extensions is not None:
            documents = [
                os.path.join(folder_path, file)
                for file in files
                if os.path.splitext(file)[1] in allowed_extensions
            ]
        else:
            documents = [os.path.join(folder_path, file) for file in files]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = random_state_validator(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            data.append(Path(filename).read_bytes())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Boxspace(
            data=data,
            filenames=filenames,
            target_names=target_names,
            target=target,
            DESCR=description,
        )

    return Boxspace(
        filenames=filenames, target_names=target_names, target=target,
        DESCR=description
    )
      
text_files_loader.__doc__ ="""\
Load text files with categories as subfolder names.

Individual samples are assumed to be files stored a two levels folder
structure such as the following::

    container_folder/
        category_1_folder/
            file1.txt
            file2.txt
            ...
            file30.txt
        category_2_folder/
            file31.txt
            file32.txt
            ...
            
The folder names are used as supervised signal label names. The individual
file names are not important.

In addition, if load_content is false it does not try to load the files in memory.
If you set load_content=True, you should also specify the encoding of the
text using the 'encoding' parameter. For many modern text files, 'utf-8'
will be the correct encoding. If you want files with a specific file extension 
(e.g. `.txt`) then you can pass a list of those file extensions to 
`allowed_extensions`.

Parameters
----------
container_path : str
    Path to the main folder holding one subfolder per category.
description : str, default=None
    A paragraph describing the characteristic of the dataset: its source,
    reference, etc.
categories : list of str, default=None
    If None (default), load all the categories. If not None, list of
    category names to load (other categories ignored).
load_content : bool, default=True
    Whether to load or not the content of the different files. If true a
    'data' attribute containing the text information is present in the data
    structure returned. If not, a filenames attribute gives the path to the
    files.
shuffle : bool, default=True
    Whether or not to shuffle the data: might be important for models that
    make the assumption that the samples are independent and identically
    distributed (i.i.d.), such as stochastic gradient descent.
encoding : str, default=None
    If None, do not try to decode the content of the files (e.g. for images
    or other non-text content). If not None, encoding to use to decode text
    files to Unicode if load_content is True.
decode_error : {'strict', 'ignore', 'replace'}, default='strict'
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. Passed as keyword
    argument 'errors' to bytes.decode.
random_state : int, RandomState instance or None, default=42
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
allowed_extensions : list of str, default=None
    List of desired file extensions to filter the files to be loaded.

Returns
-------
data : :class:`~gofast.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    data : list of str
      Only present when `load_content=True`.
      The raw text data to learn.
    target : ndarray
      The target labels (integer index).
    target_names : list
      The names of target classes.
    DESCR : str
      The full description of the dataset.
    filenames: ndarray
      The filenames holding the dataset.
"""

























