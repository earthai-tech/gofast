# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utility functions for handling, transforming, and validating 
data structures commonly used in data processing and analysis workflows. 
Functions cover a range of tasks, including data normalization, handling missing 
values, data sampling, and type validation. The module is designed to streamline 
data preparation and transformation, facilitating seamless integration into 
machine learning and statistical pipelines.
"""
import os 
import re
import copy 
import shutil
import warnings 
from typing import Any, List, Union, Dict, Optional, Set, Tuple   
from functools import reduce

import scipy 
import numpy as np 
import pandas as pd 

from .._gofastlog import gofastlog 
from ..api.types import _F, ArrayLike, NDArray, DataFrame 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, StrOptions
from ..decorators import Deprecated, RunReturn, isdf
from ..core.array_manager import to_numeric_dtypes 
from ..core.checks import ( 
    _assert_all_types, _isin, _validate_name_in, 
    is_iterable, assert_ratio
)
from ..core.io import is_data_readable 
from ..core.utils import sanitize_frame_cols 
from .validator import check_is_runned, is_frame, validate_positive_integer  

logger = gofastlog().get_gofast_logger(__name__) 

__all__= [
    'DataManager', 
    'cleaner',
    'data_extractor', 
    'extract_coordinates', 
    'nan_to_na',
    'pair_data',
    'process_and_extract_data',
    'random_sampling',
    'random_selector',
    'read_from_excelsheets',
    'read_worksheets',
    'resample_data', 
    'replace_data', 
    'long_to_wide', 
    'wide_to_long', 
    'repeat_feature_accross', 
    'merge_datasets', 
    'swap_ic'
    ]

class DataManager(BaseClass):
    """
    A class for managing and organizing files in directories.

    This class provides methods to organize files based on file types,
    name patterns, and to rename files in bulk. All operations are
    executed via the `run` method to ensure proper initialization.

    Parameters
    ----------
    root_dir : str
        The root directory containing the files to manage.

    target_dir : str
        The directory where the organized files will be placed.

    file_types : list of str, optional
        A list of file extensions to filter by (e.g., ``['.csv', '.json']``).
        If ``None``, all file types are included. Default is ``None``.

    name_patterns : list of str, optional
        A list of name patterns to filter by (e.g., ``['2023', 'report']``).
        If ``None``, all file names are included. Default is ``None``.

    move : bool, optional
        If ``True``, files are moved instead of copied. Default is ``False``.

    overwrite : bool, optional
        If ``True``, existing files at the target location will be overwritten.
        If ``False``, existing files are skipped. Default is ``False``.

    create_dirs : bool, optional
        If ``True``, missing directories in the target path are created.
        Default is ``False``.

    Attributes
    ----------
    root_dir_ : str
        The root directory containing the files to manage.

    target_dir_ : str
        The directory where the organized files will be placed.

    Methods
    -------
    run(pattern=None, replacement=None)
        Executes the data management operations.

    Examples
    --------
    >>> from gofast.tools.datautils import DataManager
    >>> manager = DataManager(
    ...     root_dir='data/raw',
    ...     target_dir='data/processed',
    ...     file_types=['.csv', '.json'],
    ...     name_patterns=['2023', 'report'],
    ...     move=True,
    ...     overwrite=True,
    ...     create_dirs=True
    ... )
    >>> manager.run(pattern='old', replacement='new')

    Notes
    -----
    The `run` method organizes files and performs renaming based on the
    initialization parameters and arguments provided to `run`.

    See Also
    --------
    shutil.move : Moves a file or directory to another location.
    shutil.copy : Copies a file to another location.
    """

    @validate_params({
        'root_dir': [str],
        'target_dir': [str],
        'file_types': [list, None],
        'name_patterns': [list, None],
        'move': [bool],
        'overwrite': [bool],
        'create_dirs': [bool]
    })
    def __init__(
        self,
        root_dir: str,
        target_dir: str,
        file_types: Optional[List[str]] = None,
        name_patterns: Optional[List[str]] = None,
        move: bool = False,
        overwrite: bool = False,
        create_dirs: bool = False
    ):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.file_types = file_types
        self.name_patterns = name_patterns
        self.move = move
        self.overwrite = overwrite
        self.create_dirs = create_dirs
    
        # Ensure root_dir exists
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Root directory '{self.root_dir}' does not exist.")

        # Create target_dir if create_dirs is True
        if self.create_dirs and not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)

        logger.debug(f"Initialized DataManager with root_dir: {self.root_dir}, "
                     f"target_dir: {self.target_dir}")

    @RunReturn
    def run(self, pattern: Optional[str] = None, replacement: Optional[str] = None):
        """
        Executes the data management operations.

        This method organizes files based on the specified filters and
        optionally renames files by replacing a pattern with a replacement
        string.

        Parameters
        ----------
        pattern : str, optional
            The pattern to search for in file names during renaming.
            If ``None``, renaming is skipped. Default is ``None``.

        replacement : str, optional
            The string to replace the pattern with during renaming.
            Required if `pattern` is provided.

        Returns
        -------
        self : DataManager
            Returns self.

        Examples
        --------
        >>> manager = DataManager(...)
        >>> manager.run(pattern='old', replacement='new')

        Notes
        -----
        The `run` method must be called before invoking any other methods.
        It sets up the necessary state for the object.
        """
        self._organize_files()
        if pattern is not None:
            if replacement is None:
                raise ValueError(
                    "Replacement string must be provided if pattern is specified.")
            self._rename_files(pattern, replacement)
        self._is_runned = True  # Mark as runned

    def get_processed_files(self) -> List[str]:
        """
        Retrieves a list of files that have been processed.

        Returns
        -------
        files : list of str
            A list of file paths that have been processed.

        Examples
        --------
        >>> manager = DataManager(...)
        >>> manager.run()
        >>> files = manager.get_processed_files()
        """
        check_is_runned(self, attributes=['_is_runned'])
        
        processed_files = []
        for dirpath, _, filenames in os.walk(self.target_dir):
            for filename in filenames:
                processed_files.append(os.path.join(dirpath, filename))
        return processed_files

    def _organize_files(self):
        """Private method to organize files based on specified filters."""
        try:
            files = self._get_filtered_files()
            for file_path in files:
                self._handle_file(file_path)
        except Exception as e:
            logger.error(f"Failed to organize files: {str(e)}")
            raise RuntimeError(f"Organizing files failed: {str(e)}") from e

    def _rename_files(self, pattern: str, replacement: str):
        """Private method to rename files by replacing a pattern."""
        try:
            for dirpath, _, filenames in os.walk(self.target_dir):
                for filename in filenames:
                    if pattern in filename:
                        old_path = os.path.join(dirpath, filename)
                        new_filename = filename.replace(pattern, replacement)
                        new_path = os.path.join(dirpath, new_filename)
                        if os.path.exists(new_path) and not self.overwrite:
                            logger.info(f"File {new_path} already exists; skipping.")
                            continue
                        os.rename(old_path, new_path)
                        logger.debug(f"Renamed {old_path} to {new_path}")
        except Exception as e:
            logger.error(f"Failed to rename files: {str(e)}")
            raise RuntimeError(f"Renaming files failed: {str(e)}") from e

    def _get_filtered_files(self):
        """Retrieves files from root_dir filtered by file_types and name_patterns."""
        matched_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if self.file_types and not any(
                        filename.endswith(ext) for ext in self.file_types):
                    continue
                if self.name_patterns and not any(
                        pat in filename for pat in self.name_patterns):
                    continue
                matched_files.append(os.path.join(dirpath, filename))
        return matched_files

    def _handle_file(self, file_path):
        """Handles moving or copying of a single file."""
        relative_path = os.path.relpath(file_path, self.root_dir)
        target_path = os.path.join(self.target_dir, relative_path)

        # Create target directories if needed
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        if os.path.exists(target_path) and not self.overwrite:
            logger.info(f"File {target_path} already exists; skipping.")
            return

        if self.move:
            shutil.move(file_path, target_path)
            logger.debug(f"Moved {file_path} to {target_path}")
        else:
            shutil.copy2(file_path, target_path)
            logger.debug(f"Copied {file_path} to {target_path}")
          
@is_data_readable 
def nan_to_na(
    data: DataFrame, 
    cat_missing_value: Optional[Union[Any,  float]] = pd.NA, 
    nan_spec: Optional[Union[Any, float]] = np.nan,
):
    """
    Converts specified NaN values in categorical columns of a pandas 
    DataFrame or Series to `pd.NA` or another specified missing value.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The input DataFrame or Series in which specified NaN values in 
        categorical columns will be converted.
        
    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used. This ensures that categorical columns 
        do not contain `np.nan` values, which can cause type 
        inconsistencies.

    nan_spec : scalar, default=np.nan
        The value that is treated as NaN in the input data. By default, 
        `np.nan` is used. This allows flexibility in specifying what is 
        considered as NaN.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The DataFrame or Series with specified NaN values in categorical 
        columns converted to the specified missing value.

    Notes
    -----
    This function ensures consistency in the representation of missing 
    values in categorical columns, avoiding issues that arise from the 
    presence of specified NaN values in such columns.

    The conversion follows the logic:
    
    .. math:: 
        \text{If column is categorical and contains `nan_spec`} 
        \rightarrow \text{Replace `nan_spec` with `cat_missing_value`}

    Examples
    --------
    >>> from gofast.tools.datautils import nan_to_na
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1.0, 2.0, np.nan], 'B': ['x', np.nan, 'z']})
    >>> df['B'] = df['B'].astype('category')
         A    B
    0  1.0    x
    1  2.0  NaN
    2  NaN    z
    
    >>> df = nan_to_na(df)
    >>> df
         A     B
    0  1.0     x
    1  2.0  <NA>
    2  NaN     z

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    pandas.Series : One-dimensional ndarray with axis labels.
    numpy.nan : IEEE 754 floating point representation of Not a Number 
        (NaN).

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.

    """
    is_frame(data, raise_exception= True, objname ='data')

    def has_nan_values(series, nan_spec):
        """Check if nan_spec exists in the series."""
        return series.isin([nan_spec]).any()
    
    if isinstance(data, pd.Series):
        if has_nan_values(data, nan_spec):
            if pd.api.types.is_categorical_dtype(data):
                data=data.astype(str)
                return data.replace({str(nan_spec): cat_missing_value})
        return data
    
    elif isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        for column in df_copy.columns:
            if has_nan_values(df_copy[column], nan_spec):
                if pd.api.types.is_categorical_dtype(df_copy[column]):
                    df_copy[column]=df_copy[column].astype(str)
                    df_copy[column] = df_copy[column].replace(
                        {str(nan_spec): cat_missing_value})
        return df_copy

def resample_data(
    *data: Any,
    samples: Union[int, float, str] = 1,
    replace: bool = False,
    random_state: int = None,
    shuffle: bool = True
) -> List[Any]:
    """
    Resample multiple data structures (arrays, sparse matrices, Series, 
    DataFrames) based on specified sample size or ratio.

    Parameters
    ----------
    *data : Any
        Variable number of array-like, sparse matrix, pandas Series, or 
        DataFrame objects to be resampled.
        
    samples : Union[int, float, str], optional
        Specifies the number of items to sample from each data structure.
        
        - If an integer greater than 1, it is treated as the exact number 
          of items to sample.
        - If a float between 0 and 1, it is treated as a ratio of the 
          total number of rows to sample.
        - If a string containing a percentage (e.g., "50%"), it calculates 
          the sample size as a percentage of the total data length.
        
        The default is 1, meaning no resampling is performed unless a 
        different value is specified.

    replace : bool, default=False
        Determines if sampling with replacement is allowed, enabling the 
        same row to be sampled multiple times.

    random_state : int, optional
        Sets the seed for the random number generator to ensure 
        reproducibility. If specified, repeated calls with the same 
        parameters will yield identical results.

    shuffle : bool, default=True
        If True, shuffles the data before sampling. Otherwise, rows are 
        selected sequentially without shuffling.

    Returns
    -------
    List[Any]
        A list of resampled data structures, each in the original format 
        (e.g., numpy array, sparse matrix, pandas DataFrame) and with the 
        specified sample size.

    Methods
    -------
    - `_determine_sample_size`: Calculates the sample size based on the 
      `samples` parameter.
    - `_perform_sampling`: Conducts the sampling process based on the 
      calculated sample size, `replace`, and `shuffle` parameters.

    Notes
    -----
    - If `samples` is given as a percentage string (e.g., "25%"), the 
      actual number of rows to sample, :math:`n`, is calculated as:
      
      .. math::
          n = \left(\frac{\text{percentage}}{100}\right) \times N

      where :math:`N` is the total number of rows in the data structure.

    - Resampling supports both dense and sparse matrices. If the input 
      contains sparse matrices stored within numpy objects, the function 
      extracts and samples them directly.

    Examples
    --------
    >>> from gofast.tools.datautils import resample_data
    >>> import numpy as np
    >>> data = np.arange(100).reshape(20, 5)

    # Resample 10 items from each data structure with replacement
    >>> resampled_data = resample_data(data, samples=10, replace=True)
    >>> print(resampled_data[0].shape)
    (10, 5)
    
    # Resample 50% of the rows from each data structure
    >>> resampled_data = resample_data(data, samples=0.5, random_state=42)
    >>> print(resampled_data[0].shape)
    (10, 5)

    # Resample data with a percentage-based sample size
    >>> resampled_data = resample_data(data, samples="25%", random_state=42)
    >>> print(resampled_data[0].shape)
    (5, 5)

    References
    ----------
    .. [1] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
           Problems", Annals of Eugenics, 1936.

    See Also
    --------
    np.random.choice : Selects random samples from an array.
    pandas.DataFrame.sample : Randomly samples rows from a DataFrame.
    """

    resampled_structures = []

    for d in data:
        # Handle sparse matrices encapsulated in numpy objects
        if isinstance(d, np.ndarray) and d.dtype == object and scipy.sparse.issparse(d.item()):
            d = d.item()  # Extract the sparse matrix from the numpy object

        # Determine sample size based on `samples` parameter
        n_samples = _determine_sample_size(d, samples, is_percent="%" in str(samples))
        
        # Sample the data structure based on the computed sample size
        sampled_d = _perform_sampling(d, n_samples, replace, random_state, shuffle)
        resampled_structures.append(sampled_d)
 
    return resampled_structures[0] if len(
        resampled_structures)==1 else resampled_structures

def _determine_sample_size(d: Any, samples: Union[int, float, str], 
                           is_percent: bool) -> int:
    """
    Determine the number of samples to draw based on the input size or ratio.
    """
    if isinstance(samples, str) and is_percent:
        samples = samples.replace("%", "")
    try:
        samples = float(samples)
    except ValueError:
        raise TypeError(f"Invalid type for 'samples': {type(samples).__name__}."
                        " Expected int, float, or percentage string.")
   
    d_length = d.shape[0] if hasattr(d, 'shape') else len(d)
    if samples < 1 or is_percent:
        return max(1, int(samples * d_length))
    return int(samples)

def _perform_sampling(d: Any, n_samples: int, replace: bool, 
                      random_state: int, shuffle: bool) -> Any:
    """
    Perform the actual sampling operation on the data structure.
    """
    if isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.sample(n=n_samples, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n_samples]
    elif scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " to CSR matrix is recommended.")
            d = d.tocsr()
        indices = np.random.choice(d.shape[0], n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d[indices]
    else:
        d_array = np.array(d) if not hasattr(d, '__array__') else d
        indices = np.random.choice(len(d_array), n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d_array[indices] if d_array.ndim == 1 else d_array[indices, :]
    

def pair_data(
    *data: Union[pd.DataFrame, List[pd.DataFrame]],
    on: Union[str, List[str]] = None,
    parse_on: bool = False,
    mode: str = 'strict',
    coerce: bool = False,
    force: bool = False,
    decimals: int = 7,
    raise_warn: bool = True
) -> pd.DataFrame:
    """
    Finds identical objects in multiple DataFrames and merges them 
    using an intersection (`cross`) strategy.

    Parameters
    ----------
    data : List[Union[pd.DataFrame, List[pd.DataFrame]]]
        A variable-length argument of pandas DataFrames for merging.
        
    on : Union[str, List[str]], optional
        Column or index level names to join on. These must exist in 
        all DataFrames. If None and `force` is False, concatenation 
        along columns axis is performed.
        
    parse_on : bool, default=False
        If True, parses `on` when provided as a string by splitting 
        it into multiple column names.
        
    mode : str, default='strict'
        Determines handling of non-DataFrame inputs. In 'strict' 
        mode, raises an error for non-DataFrame objects. In 'soft' 
        mode, ignores them.
        
    coerce : bool, default=False
        If True, truncates all DataFrames to the length of the 
        shortest DataFrame before merging.
        
    force : bool, default=False
        If True, forces `on` columns to exist in all DataFrames, 
        adding them from any DataFrame that contains them. Raises an 
        error if `on` columns are missing in all provided DataFrames.
        
    decimals : int, default=7
        Number of decimal places to round numeric `on` columns for 
        comparison. Helps ensure similar values are treated as equal.
        
    raise_warn : bool, default=True
        If True, warns user that data is concatenated along column 
        axis when `on` is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged objects based on `on` 
        columns, using cross intersection for matching.

    Methods
    -------
    - `pd.concat`: Concatenates DataFrames along columns if `on` 
      is None.
    - `pd.merge`: Merges DataFrames based on `on` columns.
    
    Notes
    -----
    - This function performs pairwise merging of DataFrames based 
      on column alignment specified in `on`.
      
    - When `decimals` is set, values in `on` columns are rounded 
      to the specified decimal places before merging to avoid 
      floating-point discrepancies:
      
      .. math::
          \text{round}(x, \text{decimals})

    - The function requires that all provided data be DataFrames if 
      `mode='strict'`. Non-DataFrame inputs in 'strict' mode raise 
      a `TypeError`.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.datautils import pair_data
    >>> data1 = pd.DataFrame({
    ...     'longitude': [110.486111],
    ...     'latitude': [26.05174],
    ...     'value': [10]
    ... })
    >>> data2 = pd.DataFrame({
    ...     'longitude': [110.486111],
    ...     'latitude': [26.05174],
    ...     'measurement': [1]
    ... })
    
    # Merge based on common columns 'longitude' and 'latitude'
    >>> pair_data(data1, data2, on=['longitude', 'latitude'], decimals=5)
       longitude  latitude  value  measurement
    0  110.48611  26.05174     10           1

    References
    ----------
    .. [1] Wes McKinney, "Data Structures for Statistical Computing 
           in Python", Proceedings of the 9th Python in Science 
           Conference, 2010.

    See Also
    --------
    pd.concat : Concatenates pandas objects along a specified axis.
    pd.merge : Merges DataFrames based on key columns.
    """

    # Filter only DataFrames if `mode` is set to 'soft'
    if str(mode).lower() == 'soft':
        d = [df for df in data if isinstance(df, pd.DataFrame)]

    # Ensure all provided data is DataFrame if `mode` is 'strict'
    is_dataframe = all(isinstance(df, pd.DataFrame) for df in d)
    if not is_dataframe:
        types = [type(df).__name__ for df in d]
        raise TypeError(f"Expected DataFrame. Got {', '.join(types)}")

    # Coerce to shortest DataFrame length if `coerce=True`
    if coerce:
        min_len = min(len(df) for df in d)
        d = [df.iloc[:min_len, :] for df in d]

    # If `on` is None and `raise_warn` is True, warn and concatenate along columns
    if on is None:
        if raise_warn:
            warnings.warn("`on` parameter is None. Performing"
                          " concatenation along columns.")
        return pd.concat(d, axis=1)

    # Parse `on` if `parse_on=True`
    if parse_on and isinstance(on, str):
        on = on.split()

    # Ensure `on` columns exist in all DataFrames if `force=True`
    if force:
        missing_cols = [col for col in on if not all(col in df.columns for df in d)]
        if missing_cols:
            d = [df.assign(**{col: d[0][col]}) for col in missing_cols 
                 for df in d if col in d[0].columns]

    # Round numeric columns in `on` columns to `decimals` if specified
    for df in d:
        for col in on:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(decimals)

    # Perform pairwise merging based on `on` columns
    data = d[0]
    for df in d[1:]:
        data = pd.merge(data, df, on=on, suffixes=('_x', '_y'))

    return data

def random_sampling(
    d,
    samples: int = None,
    replace: bool = False,
    random_state: int = None,
    shuffle: bool = True,
) -> Union[np.ndarray, 'pd.DataFrame', 'scipy.sparse.spmatrix']:
    """
    Randomly samples rows from the data, with options for shuffling, 
    sampling with replacement, and fixed randomness for reproducibility.
    
    Parameters
    ----------
    d : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to sample from. Supports any array-like structure, 
        pandas DataFrame, or scipy sparse matrix with `n_samples` 
        as rows and `n_features` as columns.

    samples : int, optional
        Number of items or ratio of items to return. If `samples` 
        is None, it defaults to 1 (selects all items). If set as 
        a float (e.g., "0.2"), it is interpreted as the percentage 
        of data to sample.
        
    replace : bool, default=False
        If True, allows sampling the same row multiple times; 
        if False, each row can only be sampled once.
        
    random_state : int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
        Controls randomness. If int or array-like, sets the seed 
        for reproducibility. If a `RandomState` or `Generator`, it 
        will be used as-is.
        
    shuffle : bool, default=True
        If True, shuffles the data before sampling; otherwise, 
        returns the top `n` samples without shuffling.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix]
        Sampled data, in the same format as `d` (array-like, sparse 
        matrix, or DataFrame) and in the shape (samples, n_features).

    Methods
    -------
    - `np.random.choice`: Selects rows randomly based on `samples` 
      and `replace` parameter.
    - `d.sample()`: Used for DataFrames to sample with more control.
    
    Notes
    -----
    - If `samples` is a string containing "%", the number of samples 
      is calculated as a percentage of the total rows:
      
      .. math::
          \text{samples} = \frac{\text{percentage}}{100} \times 
          \text{len(d)}

    - To ensure consistent sampling, especially when `replace=True`, 
      setting `random_state` is recommended for reproducibility.
    
    - The function supports various data types and automatically 
      converts `d` to a compatible structure if necessary.

    Examples
    --------
    >>> from gofast.tools.datautils import random_sampling
    >>> import numpy as np
    >>> data = np.arange(100).reshape(20, 5)
    
    # Sample 7 rows from data
    >>> random_sampling(data, samples=7).shape
    (7, 5)
    
    # Sample 10% of rows
    >>> random_sampling(data, samples="10%", random_state=42).shape
    (2, 5)

    >>> # Sampling from a pandas DataFrame with replacement
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.arange(100).reshape(20, 5))
    >>> random_sampling(df, samples=5, replace=True).shape
    (5, 5)
    
    References
    ----------
    .. [1] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
           Problems", Annals of Eugenics, 1936.

    See Also
    --------
    np.random.choice : Selects random samples from an array.
    pandas.DataFrame.sample : Randomly samples rows from a DataFrame.
    """

    # Initialize variables for calculation
    n = None
    is_percent = False
    orig = copy.deepcopy(samples)

    # Ensure data is iterable and convert if necessary
    if not hasattr(d, "__iter__"):
        d = np.array(d)

    # Set default sample size to 1 if samples is None or wildcarded
    if samples is None or str(samples) in ('1', '*'):
        samples = "100%"

    # Handle percentage-based sampling if specified as a string
    if "%" in str(samples):
        samples = str(samples).replace("%", "")
        is_percent = True
    
    # Ensure samples is a valid numerical value
    try:
        samples = float(samples)
    except ValueError:
        raise TypeError("Invalid value for 'samples'. Expected an integer "
                        f"or percentage, got {type(orig).__name__!r}")

    # Calculate the sample size based on percentage if necessary
    if samples <= 1 or is_percent:
        samples = assert_ratio(
            samples, bounds=(0, 1), exclude_values='use lower bound',
            in_percent=True
        )
        n = int(samples * (d.shape[0] if scipy.sparse.issparse(d) else len(d)))
    else:
        # Use the integer value directly
        n = int(samples)
    
    # Ensure sample size does not exceed data length
    dlen = d.shape[0] if scipy.sparse.issparse(d) else len(d)
    if n > dlen:
        n = dlen

    # Sampling for DataFrame
    if hasattr(d, 'sample'):
        return d.sample(n=n, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n, :]

    # Set random state for reproducibility
    np.random.seed(random_state)

    # Handle sparse matrix sampling
    if scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("`coo_matrix` does not support indexing. "
                          "Converting to CSR matrix for indexing.")
            d = d.tocsr()
        indices = np.random.choice(np.arange(d.shape[0]), n, replace=replace
                                   ) if shuffle else list(range(n))
        return d[indices]

    # Manage array-like data
    d = np.array(d) if not hasattr(d, '__array__') else d
    indices = np.random.choice(len(d), n, replace=replace) if shuffle else list(range(n))
    d = d[indices] if d.ndim == 1 else d[indices, :]

    return d

def read_from_excelsheets(erp_file: str) -> List[pd.DataFrame]:
    """
    Read all sheets from an Excel workbook into a list of DataFrames.

    Parameters
    ----------
    erp_file : str
        Path to the Excel file containing multiple sheets.

    Returns
    -------
    List[DataFrame]
        A list where the first element is the file base name and subsequent 
        elements are DataFrames of each sheet.

    Examples
    --------
    >>> read_from_excelsheets("data.xlsx")
    ['data', DataFrame1, DataFrame2, ...]

    Notes
    -----
    This function reads all sheets at once, which is helpful for files with 
    structured data across multiple sheets.
    """
    all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(erp_file, sheet_name=None)
    file_base_name = os.path.basename(os.path.splitext(erp_file)[0])
    list_of_df = [file_base_name] + [pd.DataFrame(values) for values in all_sheets.values()]

    return list_of_df


def read_worksheets(
        *data: str
        ) -> Tuple[Optional[List[pd.DataFrame]], Optional[List[str]]]:
    """
    Reads all `.xlsx` sheets from given file paths or directories and returns
    the contents as DataFrames along with sheet names. 
    This function processes each sheet in a workbook as a separate DataFrame
    and collects all sheet names, replacing special characters in names with
    underscores.

    Parameters
    ----------
    data : str
        Variable-length argument of file paths or directories. 
        Each path should be a string pointing to an `.xlsx` file or a directory
        containing `.xlsx` files. Only files with the `.xlsx` extension are 
        read; other file types will raise an error.

    Returns
    -------
    Tuple[Optional[List[pd.DataFrame]], Optional[List[str]]]
        A tuple containing two elements:
        
        - A list of DataFrames, each representing a sheet from the specified
        `.xlsx` files.
        - A list of sheet names corresponding to the DataFrames. 
          Special characters in sheet names are replaced with underscores 
          for standardization.

    Methods
    -------
    - `pd.read_excel(d, sheet_name=None)`: Reads Excel file and loads all 
      sheets into a dictionary of DataFrames.
    - `re.sub`: Replaces special characters in sheet names with 
      underscores.

    Notes
    -----
    - If a directory is provided, the function searches for all 
      `.xlsx` files in that directory. Only Excel files with the 
      `.xlsx` extension are supported.
      
    - If no `.xlsx` files are found in the provided paths, the 
      function returns `(None, None)`.

    - To maintain consistency in sheet names, special characters 
      in sheet names are replaced with underscores using 
      :class:`re.Regex`.

    - Mathematically, if :math:`d` represents an Excel file in 
      `data`, then:
      
      .. math::
          \text{dataframes}_{d} = f(\text{Excel file})
      
      where :math:`f` denotes loading all sheets within an Excel 
      file into separate DataFrames.

    Examples
    --------
    >>> from gofast.tools.datautils import read_worksheets
    >>> # Example 1: Reading a single Excel file
    >>> file_path = r'F:/repositories/gofast/data/erp/sheets/gbalo.xlsx'
    >>> data, sheet_names = read_worksheets(file_path)
    >>> sheet_names
    ['l11', 'l10', 'l02']
    
    >>> # Example 2: Reading all .xlsx files in a directory
    >>> import os
    >>> dir_path = os.path.dirname(file_path)
    >>> data, sheet_names = read_worksheets(dir_path)
    >>> sheet_names
    ['l11', 'l10', 'l02', 'l12', 'l13']

    References
    ----------
    .. [1] McKinney, Wes, *Data Structures for Statistical Computing 
           in Python*, Proceedings of the 9th Python in Science 
           Conference, 2010.

    See Also
    --------
    pd.read_excel : Reads an Excel file and returns DataFrames 
                    of all sheets.
    os.path.isdir : Checks if the path is a directory.
    os.path.isfile : Checks if the path is a file.
    """

    # Temporary list to store valid .xlsx files
    dtem = []
    data = [o for o in data if isinstance(o, str)]

    # Iterate over each path provided in data
    for o in data:
        if os.path.isdir(o):
            # Get all files in the directory, filtering for .xlsx files
            dlist = os.listdir(o)
            p = [os.path.join(o, f) for f in dlist if f.endswith('.xlsx')]
            dtem.extend(p)
        elif os.path.isfile(o):
            # Check if the file is an .xlsx file
            _, ex = os.path.splitext(o)
            if ex == '.xlsx':
                dtem.append(o)

    # Deep copy of the collected .xlsx files
    data = copy.deepcopy(dtem)

    # Return None if no valid Excel files are found
    if len(data) == 0:
        return None, None

    # Dictionary to store DataFrames by sheet name
    ddict = {}
    regex = re.compile(r'[$& #@%^!]', flags=re.IGNORECASE)

    # Read each Excel file and store sheets
    for d in data:
        try:
            ddict.update(**pd.read_excel(d, sheet_name=None))
        except Exception:
            pass  # Continue if any file fails to read

    # Raise error if no data could be read
    if len(ddict) == 0:
        raise TypeError("No readable data found in the provided paths.")

    # Standardize sheet names and store them
    sheet_names = list(map(lambda o: regex.sub('_', o).lower(), ddict.keys()))

    # Collect the DataFrames
    data = list(ddict.values())

    return data, sheet_names

def process_and_extract_data(
    *args: ArrayLike, 
    columns: Optional[List[Union[str, int]]] = None,
    enforce_extraction: bool = True, 
    allow_split: bool = False, 
    search_multiple: bool = False,
    ensure_uniform_length: bool = False, 
    to_array: bool = False,
    on_error: str = 'raise',
) -> List[np.ndarray]:
    """
    Extracts and processes data from various input types, focusing on column 
    extraction from pandas DataFrames and conversion of inputs to numpy 
    arrays or pandas Series.

    Parameters
    ----------
    *args : ArrayLike
        A variable number of inputs, each can be a list, numpy array, pandas 
        Series,dictionary, or pandas DataFrame.
    columns : List[Union[str, int]], optional
        Specific columns to extract from pandas DataFrames. If not provided, 
        the function behaves differently based on `allow_split`.
    enforce_extraction : bool, default=True
        Forces the function to try extracting `columns` from DataFrames. 
        If False, DataFrames are returned without column extraction unless 
        `allow_split` is True.
        Removing non-conforming elements if True.
    allow_split : bool, default=False
        If True and a DataFrame is provided without `columns`, splits the 
        DataFrame into its constituent columns.
    search_multiple : bool, default=False
        Allows searching for `columns` across multiple DataFrame inputs. Once 
        a column is found, it is not searched for in subsequent DataFrames.
    ensure_uniform_length : bool, default=False
        Checks that all extracted arrays have the same length. Raises an error
        if they don't.
    to_array : bool, default=False
        Converts all extracted pandas Series to numpy arrays.
    on_error : str, {'raise', 'ignore'}, default='raise'
        Determines how to handle errors during column extraction or when 
        enforcing uniform length. 'raise' will raise an error, 'ignore' will 
        skip the problematic input.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays or pandas Series extracted based 
        on the specified conditions.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.datautils import process_and_extract_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> process_and_extract_data(data, columns=['A'], to_array=True)
    [array([1, 2, 3])]

    Splitting DataFrame into individual arrays:

    >>> process_and_extract_data(data, allow_split=True, to_array=True)
    [array([1, 2, 3]), array([4, 5, 6])]

    Extracting columns from multiple DataFrames:

    >>> data2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
    >>> process_and_extract_data(data, data2, columns=['A', 'C'], 
                                  search_multiple=True, to_array=True)
    [array([1, 2, 3]), array([7, 8, 9])]

    Handling mixed data types:

    >>> process_and_extract_data([1, 2, 3], {'E': [13, 14, 15]}, to_array=True)
    [array([1, 2, 3]), array([13, 14, 15])]
    
    Extracting columns from multiple DataFrames and enforcing uniform length:
    >>> data2 = pd.DataFrame({'C': [7, 8, 9, 10], 'D': [11, 12, 13, 14]})
    >>> result = process_and_extract_data(
        data, data2, columns=['A', 'C'],search_multiple=True,
        ensure_uniform_length=True, to_array=True)
    ValueError: Extracted data arrays do not have uniform length.
    """
    extracted_data = []
    columns_found: Set[Union[str, int]] = set()

    def _process_input(
            input_data: ArrayLike,
            target_columns: Optional[List[Union[str, int]]], 
            to_array: bool) -> Optional[np.ndarray]:
        """
        Processes each input based on its type, extracting specified columns 
        if necessary, and converting to numpy array if specified.
        """
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
            return input_data if len(input_data.shape
                                     ) == 1 or not enforce_extraction else None

        elif isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)

        if isinstance(input_data, pd.DataFrame):
            if target_columns:
                for col in target_columns:
                    if col in input_data.columns and (
                            search_multiple or col not in columns_found):
                        data_to_add = input_data[col].to_numpy(
                            ) if to_array else input_data[col]
                        extracted_data.append(data_to_add)
                        columns_found.add(col)
                    elif on_error == 'raise':
                        raise ValueError(f"Column {col} not found in DataFrame.")
            elif allow_split:
                for col in input_data.columns:
                    data_to_add = input_data[col].to_numpy(
                        ) if to_array else input_data[col]
                    extracted_data.append(data_to_add)
            return None

        if isinstance(input_data, np.ndarray):
            if input_data.ndim > 1 and allow_split:
                input_data = np.hsplit(input_data, input_data.shape[1])
                for arr in input_data:
                    extracted_data.append(arr.squeeze())
                return None
            elif input_data.ndim > 1 and enforce_extraction and on_error == 'raise':
                raise ValueError("Multidimensional array found while "
                                 "`enforce_extraction` is True.")
            return input_data if to_array else np.squeeze(input_data)

        return input_data.to_numpy() if to_array and isinstance(
            input_data, pd.Series) else input_data

    for arg in args:
        result = _process_input(arg, columns, to_array)
        if result is not None:
            extracted_data.append(result)

    if ensure_uniform_length and not all(len(x) == len(
            extracted_data[0]) for x in extracted_data):
        if on_error == 'raise':
            raise ValueError(
                "Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def random_selector(
    arr: ArrayLike,
    value: Union[float, ArrayLike, str],
    seed: int = None,
    shuffle: bool = False
) -> np.ndarray:
    """
    Randomly select specified values from an array, using a value 
    count, percentage, or subset. Provides consistent selection if 
    seeded, and can shuffle the result.

    Parameters
    ----------
    arr : ArrayLike
        Input array of values from which selections are made. 
        Accepts any array-like structure (e.g., list, ndarray) 
        for processing.
        
    value : Union[float, ArrayLike, str]
        Specifies the number or subset of values to select.
        
        - If `value` is a float, it is interpreted as the number 
          of items to select from `arr`.
        - If `value` is an array-like, it indicates the exact 
          values to select, provided they exist within `arr`.
        - If `value` is a string containing a percentage 
          (e.g., `"50%"`), it calculates the proportion to select 
          based on the length of `arr`, given by:
          
          .. math::
              \text{value} = \left( \frac{\text{percentage}}{100} \right) \times \text{len(arr)}
          
    seed : int, optional
        Seed for the random number generator, which ensures 
        repeatable selections when set. Defaults to None.
        
    shuffle : bool, default=False
        If True, shuffles the selected values after extraction.
        
    Returns
    -------
    np.ndarray
        Array containing the randomly selected values from `arr`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.datautils import random_selector
    >>> data = np.arange(42)
    
    # Select 7 elements deterministically using a seed
    >>> random_selector(data, 7, seed=42)
    array([0, 1, 2, 3, 4, 5, 6])
    
    # Select specific values present in the array
    >>> random_selector(data, (23, 13, 7))
    array([ 7, 13, 23])
    
    # Select a percentage of values
    >>> random_selector(data, "7%", seed=42)
    array([0, 1])
    
    # Select 70% of values with shuffling enabled
    >>> random_selector(data, "70%", seed=42, shuffle=True)
    array([ 0,  5, 20, 25, 13,  7, 22, 10, 12, 27, 23, 21, 16,  3,  1, 17,  8,
            6,  4,  2, 19, 11, 18, 24, 14, 15,  9, 28, 26])

    Notes
    -----
    - The `value` parameter can be a float representing the count, 
      a string containing a percentage, or an array of elements to 
      select. For invalid types, the function raises a `TypeError`.
      
    - The `seed` parameter is essential for reproducibility. 
      When set, repeated calls with the same parameters will yield 
      identical results.
      
    - This function is helpful for sampling data subsets in 
      machine learning and statistical analysis.
      
    References
    ----------
    .. [1] Fisher, Ronald A., "The Use of Multiple Measurements in 
           Taxonomic Problems", Annals of Eugenics, 1936.

    See Also
    --------
    numpy.random.permutation : Randomly permutes elements in an array.
    numpy.random.shuffle : Shuffles array in place.
    """
    
    # Error message for invalid input
    msg = "Non-numerical value is not allowed. Got {!r}."
    
    # Set seed if provided for reproducibility
    if seed is not None:
        seed = _assert_all_types(seed, int, float, objname='Seed')
        np.random.seed(seed)
    
    # Deep copy of value for error reporting if necessary
    v = copy.deepcopy(value)
    
    # If value is not iterable (excluding strings), convert to string
    if not is_iterable(value, exclude_string=True):
        value = str(value)
        
        # Handle percentage-based selection
        if '%' in value:
            try:
                value = float(value.replace('%', '')) / 100
            except:
                raise TypeError(msg.format(v))
            # Calculate number of items to select based on percentage
            value *= len(arr)
        
        try:
            # Convert value to integer if possible
            value = int(value)
        except:
            raise TypeError(msg.format(v))
    
        # Ensure the selected count does not exceed array length
        if value > len(arr):
            raise ValueError(f"Number {value} is out of range. "
                             f"Expected value less than {len(arr)}.")
        
        # Randomly select `value` items
        value = np.random.permutation(value)
        
    # Ensure `arr` is array-like and flatten if multi-dimensional
    arr = np.array(is_iterable(arr, exclude_string=True, transform=True))
    arr = arr.ravel() if arr.ndim != 1 else arr
        
    # Select specified elements in `value`
    mask = _isin(arr, value, return_mask=True)
    arr = arr[mask]
    
    # Shuffle the array if specified
    if shuffle:
        np.random.shuffle(arr)
    
    return arr

@is_data_readable 
def cleaner(
    data: Union[DataFrame, NDArray],
    columns: List[str] = None,
    inplace: bool = False,
    labels: List[Union[int, str]] = None,
    func: _F = None,
    mode: str = 'clean',
    **kws
) -> Union[DataFrame, NDArray, None]:
    """
    Sanitize data by dropping specified labels from rows or columns 
    with optional column transformation. This function allows both 
    structured data (e.g., pandas DataFrame) and unstructured 2D array 
    formats, applying universal cleaning functions if provided. 

    Parameters
    ----------
    data : Union[pd.DataFrame, NDArray]
        Data structure to process, supporting either a 
        :class:`pandas.DataFrame` or a 2D :class:`numpy.ndarray`.
        If a numpy array is passed, it will be converted to a 
        DataFrame internally to facilitate label-based operations. 
        
    columns : List[str], optional
        List of column labels to operate on, by default None.
        If specified, the columns matching these labels will be 
        subject to any transformations or deletions specified by 
        `mode`. This is useful when targeting specific columns 
        without altering others.
        
    inplace : bool, default=False
        If True, modifies `data` directly; if False, returns a 
        new DataFrame or array with modifications. Note that when 
        `data` is initially provided as an array, this parameter 
        is overridden to False to ensure consistent return types.
        
    labels : List[Union[int, str]], optional
        Index or column labels to drop. Can be a list of column 
        names or index labels. If provided, only the specified 
        labels will be targeted for removal or transformation.
        
    func : Callable, optional
        Universal cleaning function to apply to the columns 
        (e.g., string cleaning, handling missing values).
        If `mode='clean'`, `func` will be applied to specified 
        columns, allowing customized data preprocessing.
        
    mode : str, default='clean'
        Operational mode controlling function behavior. Supported 
        options are:
            - 'clean': Applies the `func` callable to columns for 
              preprocessing tasks.
            - 'drop': Removes rows or columns based on `labels` or 
              `columns`. Follows similar behavior to 
              :func:`pandas.DataFrame.drop`.
              
    **kws : dict
        Additional keyword arguments passed to :func:`pandas.DataFrame.drop`
        when `mode='drop'`. Allows configuration of drop operation 
        (e.g., `axis`, `errors`).
        
    Returns
    -------
    Union[pd.DataFrame, NDArray, None]
        Returns the cleaned or transformed DataFrame or array, 
        depending on the input type. If `inplace=True`, returns None. 
        If the original data was an array, the output remains an array.
        
    Methods
    -------
    - `sanitize_frame_cols(data, inplace, func)`: Performs cleaning 
      on columns based on a function.
    - `to_numeric_dtypes(data)`: Ensures that all applicable data 
      types are converted to numeric types, which can be essential 
      for computational consistency.
      
    Notes
    -----
    - By default, when a 2D array is provided as `data`, it is 
      converted to a DataFrame for processing purposes and then 
      returned as an array after operations are complete. This 
      ensures compatibility with label-based operations.
      
    - The primary operations in this function can be mathematically 
      described as follows:
      
      .. math::
          \text{DataFrame}_{\text{cleaned}} = 
          f(\text{DataFrame}_{\text{original}})
          
      where :math:`f` is a transformation applied by `func` when 
      `mode='clean'`, and a subset selection based on `labels` 
      otherwise.

    Examples
    --------
    >>> from gofast.tools.datautils import cleaner
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, None, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> # Example: Clean using a lambda function
    >>> cleaner(data, columns=['B'], func=lambda x: x.fillna(0), mode='clean')
    
    >>> # Example: Drop rows with labels [0, 2]
    >>> cleaner(data, labels=[0, 2], mode='drop', inplace=True)

    References
    ----------
    .. [1] McKinney, Wes, *Data Structures for Statistical Computing in Python*, 
           Proceedings of the 9th Python in Science Conference, 2010.

    See Also
    --------
    pd.DataFrame.drop : Removes specified labels from rows or columns.
    """
    # Validate and set mode operation.
    mode = _validate_name_in(
        mode, defaults=("drop", 'clean'), expect_name='drop'
    )
    
    if mode == 'clean':
        # If mode is clean, apply column transformations.
        return sanitize_frame_cols(data, inplace=inplace, func=func)
    
    objtype = 'array'
    if not hasattr(data, '__array__'):
        # Convert to numpy array if not array-like
        data = np.array(data)
    
    # Determine object type for handling pandas data.
    if hasattr(data, "columns"):
        objtype = "pd"
    
    if objtype == 'array':
        # Convert numpy array to DataFrame for label-based processing.
        data = pd.DataFrame(data)
        inplace = False  # Disable inplace for numpy output

    # Process columns if specified
    if columns is not None:
        columns = is_iterable(
            columns, exclude_string=True,
            parse_string=True, transform=True
        )
    
    # Perform drop operation on DataFrame
    data = data.drop(labels=labels, columns=columns, inplace=inplace, **kws)
    
    # Convert all applicable types to numeric types for consistency
    data = to_numeric_dtypes(data)
    
    # Return as numpy array if original input was array-like
    return np.array(data) if objtype == 'array' else data

@is_data_readable
def data_extractor(
    data: pd.DataFrame,
    columns: Union[str, List[str]] = None,
    as_frame: bool = False,
    drop_columns: bool = False,
    default_columns: List[Tuple[str, str]] = None,
    raise_exception: Union[bool, str] = True,
    verbose: int = 0,
    round_decimals: int = None,
    fillna_value: Any = None,
    unique: bool = False,
    coerce_dtype: Any = None
) -> Tuple[Union[Tuple[float, float], pd.DataFrame, None],
           pd.DataFrame, Tuple[str, ...]]:
    """
    Extracts specified columns (e.g., coordinates) from a DataFrame, with options 
    for formatting, dropping, rounding, and unique selection.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame expected to contain specified columns, such as 
        `longitude`/`latitude` or `easting`/`northing` for coordinates.

    columns : Union[str, List[str]], optional
        Column(s) to extract. If `None`, attempts to detect default columns 
        based on `default_columns`.

    as_frame : bool, default=False
        If True, returns extracted columns as a DataFrame. If False, computes 
        and returns the midpoint values for coordinates.

    drop_columns : bool, default=False
        If True, removes extracted columns from `data` after extraction.
        
    default_columns : List[Tuple[str, str]], optional
        List of default column pairs to search for if `columns` is `None`. 

    default_columns : List[Tuple[str, str]], optional
        List of tuples specifying default column pairs to search for in `data` 
        if `columns` is not provided. For example, 
        `[('longitude', 'latitude'), ('easting', 'northing')]`. If no matches 
        are found and `columns` is `None`, raises an error or warning based 
        on `raise_exception`. If `None`, no default columns are assumed.

    raise_exception : Union[bool, str], default=True
        If True, raises an error if `data` is not a DataFrame or columns are 
        missing. If False, converts errors to warnings. If set to `"mute"` or 
        `"silence"`, suppresses warnings entirely.

    verbose : int, default=0
        If greater than 0, outputs messages about detected columns and 
        transformations.

    round_decimals : int, optional
        If specified, rounds extracted column values to the given number of 
        decimal places.

    fillna_value : Any, optional
        If specified, fills missing values in extracted columns with 
        `fillna_value`.

    unique : bool, default=False
        If True, returns only unique values in extracted columns.

    coerce_dtype : Any, optional
        If specified, coerces extracted column(s) to the provided data type.

    Returns
    -------
    Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, ...]]
        - The extracted data as either the midpoint tuple or DataFrame, 
          depending on `as_frame`.
        - The modified original DataFrame, with extracted columns optionally 
          removed.
        - A tuple of detected column names or an empty tuple if none are 
          detected.

    Notes
    -----
    - If `as_frame=False`, computes the midpoint of coordinates by averaging 
      the values:
      
      .. math::
          \text{midpoint} = \left(\frac{\sum \text{longitudes}}{n}, 
          \frac{\sum \text{latitudes}}{n}\right)

    - If `fillna_value` is specified, missing values in extracted columns 
      are filled before further processing.

    Examples
    --------
    >>> import gofast as gf
    >>> from gofast.tools.datautils import data_extractor
    >>> testdata = gf.datasets.make_erp(n_stations=7, seed=42).frame

    # Extract longitude/latitude midpoint
    >>> xy, modified_data, columns = data_extractor(testdata)
    >>> xy, columns
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))

    # Extract as DataFrame and round coordinates
    >>> xy, modified_data, columns = data_extractor(testdata, as_frame=True, round_decimals=3)
    >>> xy.head(2)
       longitude  latitude
    0    110.486    26.051
    1    110.486    26.051

    # Extract specific columns with unique values and drop from DataFrame
    >>> xy, modified_data, columns = data_extractor(
        testdata, columns=['station', 'resistivity'], unique=True, drop_columns=True)
    >>> xy, modified_data.head(2)
    (array([[0.0, 1.0], [20.0, 167.5]]), <DataFrame without 'station' and 'resistivity'>)

    References
    ----------
    .. [1] Fotheringham, A. Stewart, *Geographically Weighted Regression: 
           The Analysis of Spatially Varying Relationships*, Wiley, 2002.

    See Also
    --------
    pd.DataFrame : Main pandas data structure for handling tabular data.
    np.nanmean : Computes the mean along specified axis, ignoring NaNs.
    """

    def validate_columns(d: pd.DataFrame, cols: List[str]) -> List[str]:
        """Check if columns exist in DataFrame, raising or warning if not."""
        missing = [col for col in cols if col not in d.columns]
        if missing:
            msg = f"Columns {missing} not found in DataFrame."
            if str(raise_exception).lower() == 'true':
                raise KeyError(msg)
            elif raise_exception not in ('mute', 'silence'):
                warnings.warn(msg)
        return [col for col in cols if col in d.columns]

    # Validate input DataFrame
    if not isinstance(data, pd.DataFrame):
        emsg = f"Expected a DataFrame but got {type(data).__name__!r}."
        if str(raise_exception).lower() == 'true':
            raise TypeError(emsg)
        elif raise_exception not in ('mute', 'silence'):
            warnings.warn(emsg)
        return None, data, ()

    # Determine columns to extract based on user input or defaults
    if columns is None:
        if default_columns is not None:
            for col_pair in default_columns:
                if all(col in data.columns for col in col_pair):
                    columns = list(col_pair)
                    break
        if columns is None:
            if str(raise_exception).lower() == 'true':
                raise ValueError("No default columns found in DataFrame.")
            if raise_exception not in ('mute', 'silence'):
                warnings.warn("No default columns found in DataFrame.")
            return None, data, ()

    # Validate extracted columns
    columns = validate_columns(data, columns)

    # Extract specified columns
    extracted = data[columns].copy()
    
    # Apply optional transformations
    if fillna_value is not None:
        extracted.fillna(fillna_value, inplace=True)
    if unique:
        extracted = extracted.drop_duplicates()
    if coerce_dtype:
        extracted = extracted.astype(coerce_dtype)
    if round_decimals is not None:
        extracted = extracted.round(round_decimals)

    # Compute midpoint if `as_frame=False`
    extracted_data = extracted if as_frame else tuple(
        np.nanmean(extracted.values, axis=0))

    # Drop columns from original DataFrame if `drop_columns=True`
    if drop_columns:
        data.drop(columns=columns, inplace=True)

    # Display verbose messages if enabled
    if verbose > 0:
        print("### Extracted columns:", columns)
        if drop_columns:
            print("### Dropped columns from DataFrame.")

    return extracted_data, data, tuple(columns)

def extract_coordinates(
    d: pd.DataFrame,
    as_frame: bool = False,
    drop_xy: bool = False,
    raise_exception: Union[bool, str] = True,
    verbose: int = 0
) -> Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]:
    """
    Identifies coordinate columns (longitude/latitude or easting/northing) 
    in a DataFrame, returns the coordinates or their central values, and 
    optionally removes the coordinate columns from the DataFrame.

    Parameters
    ----------
    d : pd.DataFrame
        The DataFrame expected to contain coordinates (`longitude` and 
        `latitude` or `easting` and `northing`). If both types are present, 
        `longitude` and `latitude` are prioritized.

    as_frame : bool, default=False
        If True, returns the coordinate columns as a DataFrame. If False, 
        computes and returns the midpoint values.

    drop_xy : bool, default=False
        If True, removes coordinate columns (`longitude`/`latitude` or 
        `easting`/`northing`) from the DataFrame after extracting them.

    raise_exception : Union[bool, str], default=True
        If True, raises an error if `d` is not a DataFrame. If set to False, 
        converts errors to warnings. If set to "mute" or "silence", suppresses 
        warnings.

    verbose : int, default=0
        If greater than 0, outputs messages about coordinate detection.

    Returns
    -------
    Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]
        - A tuple containing either the midpoint (longitude, latitude) or 
          (easting, northing) if `as_frame=False` or the coordinate columns 
          as a DataFrame if `as_frame=True`.
        - The original DataFrame, optionally with coordinates removed if 
          `drop_xy=True`.
        - A tuple of detected coordinate column names, or an empty tuple if 
          none are detected.

    Notes
    -----
    - This function searches for either `longitude`/`latitude` or 
      `easting`/`northing` columns and returns them as coordinates. If both 
      are found, `longitude`/`latitude` is prioritized.
      
    - To calculate the midpoint of the coordinates, the function averages 
      the values in the columns:

      .. math::
          \text{midpoint} = \left(\frac{\text{longitude}_{min} + \text{longitude}_{max}}{2}, 
          \frac{\text{latitude}_{min} + \text{latitude}_{max}}{2}\right)

    Examples
    --------
    >>> import gofast as gf
    >>> from gofast.tools.datautils import extract_coordinates
    >>> testdata = gf.datasets.make_erp(n_stations=7, seed=42).frame

    # Extract midpoint coordinates
    >>> xy, d, xynames = extract_coordinates(testdata)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))

    # Extract coordinates as a DataFrame without removing columns
    >>> xy, d, xynames = extract_coordinates(testdata, as_frame=True)
    >>> xy.head(2)
       longitude   latitude
    0  110.485833  26.051389
    1  110.485982  26.051577

    # Drop coordinate columns from the DataFrame
    >>> xy, d, xynames = extract_coordinates(testdata, drop_xy=True)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    References
    ----------
    .. [1] Fotheringham, A. Stewart, *Geographically Weighted Regression: 
           The Analysis of Spatially Varying Relationships*, Wiley, 2002.

    See Also
    --------
    pd.DataFrame : Main pandas data structure for handling tabular data.
    np.nanmean : Computes the mean along specified axis, ignoring NaNs.
    """
    
    def rename_if_exists(val: str, col: pd.Index, default: str) -> pd.DataFrame:
        """Rename column in `d` if `val` is found in column names."""
        match = list(filter(lambda x: val in x.lower(), col))
        if match:
            d.rename(columns={match[0]: default}, inplace=True)
        return d

    # Validate input is a DataFrame
    if not (hasattr(d, 'columns') and hasattr(d, '__array__')):
        emsg = ("Expected a DataFrame containing coordinates (`longitude`/"
                "`latitude` or `easting`/`northing`). Got type: "
                f"{type(d).__name__!r}")
        
        raise_exception = str(raise_exception).lower().strip()
        if raise_exception == 'true':
            raise TypeError(emsg)
        if raise_exception not in ('mute', 'silence'):
            warnings.warn(emsg)
        return None, d, ()

    # Rename columns to standardized names if they contain coordinate values
    for name, std_name in zip(['lat', 'lon', 'east', 'north'], 
                              ['latitude', 'longitude', 'easting', 'northing']):
        d = rename_if_exists(name, d.columns, std_name)

    # Check for and prioritize coordinate columns
    coord_columns = []
    for x, y in [('longitude', 'latitude'), ('easting', 'northing')]:
        if x in d.columns and y in d.columns:
            coord_columns = [x, y]
            break

    # Extract coordinates as DataFrame or midpoint
    if coord_columns:
        xy = d[coord_columns] if as_frame else tuple(
            np.nanmean(d[coord_columns].values, axis=0))
    else:
        xy = None
    
    # Drop coordinates if `drop_xy=True`
    if drop_xy and coord_columns:
        d.drop(columns=coord_columns, inplace=True)

    # Verbose messaging
    if verbose > 0:
        print("###", "No" if not coord_columns else coord_columns, "coordinates found.")
    
    return xy, d, tuple(coord_columns)

@Deprecated(reason=( 
    "This function is deprecated and will be removed in future versions. "
    "Please use `extract_coordinates` instead, which provides enhanced "
    "flexibility and robustness for coordinate extraction.")
)
def get_xy_coordinates(
        d, as_frame=False, drop_xy=False, raise_exception=True, verbose=0
    ):
    """Check whether the coordinate values x, y exist in the data.
    
    Parameters 
    ------------
    d: Dataframe 
       Frame that is expected to contain the longitude/latitude or 
       easting/northing coordinates.  Note if all types of coordinates are
       included in the data frame, the longitude/latitude takes the 
       priority. 
    as_frame: bool, default= False, 
       Returns the coordinates values if included in the data as a frame rather 
       than computing the middle points of the line 
    drop_xy: bool, default=False, 
       Drop the coordinates in the data and return the data transformed inplace 
       
    raise_exception: bool, default=True 
       raise error message if data is not a dataframe. If set to ``False``, 
       exception is converted to a warning instead. To mute the warning set 
       `raise_exception` to ``mute``
       
    verbose: int, default=0 
      Send message whether coordinates are detected. 
         
    returns 
    --------
    xy, d, xynames: Tuple 
      xy : tuple of float ( longitude, latitude) or (easting/northing ) 
         if `as_frame` is set to ``True``. 
      d: Dataframe transformed (coordinated removed )  or not
      xynames: str, the name of coordinates detected. 
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.datautils import get_xy_coordinates 
    >>> testdata = gf.make_erp ( n_stations =7, seed =42 ).frame 
    >>> xy, d, xynames = get_xy_coordinates ( testdata,  )
    >>> xy , xynames 
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
        longitude   latitude        easting      northing
    0  110.485833  26.051389  448565.380621  2.881476e+06
    1  110.485982  26.051577  448580.339199  2.881497e+06
    >>> # remove longitude and  lat in data 
    >>> testdata = testdata.drop (columns =['longitude', 'latitude']) 
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
             easting      northing
    0  448565.380621  2.881476e+06
    1  448580.339199  2.881497e+06
    >>> # note testdata should be transformed inplace when drop_xy is set to True
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    ((448610.25612032827, 2881538.4380570543), ('easting', 'northing'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> testdata.head(2) # coordinates are henceforth been dropped 
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    (None, ())
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    """   
    
    def get_value_in ( val,  col , default): 
        """ Get the value in the frame columns if `val` exists in """
        x = list( filter ( lambda x: x.find (val)>=0 , col)
                   )
        if len(x) !=0: 
            # now rename col  
            d.rename (columns = {x[0]: str(default) }, inplace = True ) 
            
        return d

    if not (
            hasattr ( d, 'columns') and hasattr ( d, '__array__') 
            ) : 
        emsg = ("Expect dataframe containing coordinates longitude/latitude"
                f" or easting/northing. Got {type (d).__name__!r}")
        
        raise_exception = str(raise_exception).lower().strip() 
        if raise_exception=='true': 
            raise TypeError ( emsg )
        
        if raise_exception  not in ('mute', 'silence'):  
            warnings.warn( emsg )
       
        return d 
    
    # check whether coordinates exists in the data columns 
    for name, tname in zip ( ('lat', 'lon', 'east', 'north'), 
                     ( 'latitude', 'longitude', 'easting', 'northing')
                     ) : 
        d = get_value_in(name, col = d.columns , default = tname )
       
    # get the exist coodinates 
    coord_columns  = []
    for x, y in zip ( ( 'longitude', 'easting' ), ( 'latitude', 'northing')):
        if ( x  in d.columns and y in d.columns ): 
            coord_columns.extend  ( [x, y] )

    xy  = d[ coord_columns] if len(coord_columns)!=0 else None 

    if ( not as_frame 
        and xy is not None ) : 
        # take the middle of the line and if both types of 
        # coordinates are supplied , take longitude and latitude 
        # and drop easting and northing  
        xy = tuple ( np.nanmean ( np.array ( xy ) , axis =0 )) [:2]

    xynames = tuple ( coord_columns)[:2]
    if ( 
            drop_xy  and len( coord_columns) !=0
            ): 
        # modifie the data inplace 
        d.drop ( columns=coord_columns, inplace = True  )

    if verbose: 
        print("###", "No" if len(xynames)==0 else ( 
            tuple (xy.columns) if as_frame else xy), "coordinates found.")
        
    return  xy , d , xynames 
       
def replace_data(
    X:Union [np.ndarray, pd.DataFrame], 
    y: Union [np.ndarray, pd.Series] = None, 
    n: int = 1, 
    axis: int = 0, 
    reset_index: bool = False,
    include_original: bool = False,
    random_sample: bool = False,
    shuffle: bool = False
) -> Union [ np.ndarray, pd.DataFrame , Tuple[
    np.ndarray , pd.DataFrame, np.ndarray, pd.Series]]:
    """
    Duplicates the data `n` times along a specified axis and applies various 
    optional transformations to augment the data suitability for further 
    processing or analysis.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        The input data to process. Sparse matrices are not supported.
    y : Optional[Union[np.ndarray, pd.Series]], optional
        Additional target data to process alongside `X`. Default is None.
    n : int, optional
        The number of times to replicate the data. Default is 1.
    axis : int, optional
        The axis along which to concatenate the data. Default is 0.
    reset_index : bool, optional
        If True and `X` is a DataFrame, resets the index without adding
        the old index as a column. Default is False.
    include_original : bool, optional
        If True, the original data is included in the output alongside
        the replicated data. Default is False.
    random_sample : bool, optional
        If True, samples from `X` randomly with replacement. Default is False.
    shuffle : bool, optional
        If True, shuffles the concatenated data. Default is False.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, Tuple[Union[np.ndarray, pd.DataFrame], 
                                          Union[np.ndarray, pd.Series]]]
        The augmented data, either as a single array or DataFrame, or as a tuple
        of arrays/DataFrames if `y` is provided.

    Notes
    -----
    The replacement is mathematically formulated as follows:
    Let :math:`X` be a dataset with :math:`m` elements. The function replicates 
    :math:`X` `n` times, resulting in a new dataset :math:`X'` of :math:`m * n` 
    elements if `include_original` is False. If `include_original` is True,
    :math:`X'` will have :math:`m * (n + 1)` elements.

    Examples
    --------
    
    >>> import numpy as np 
    >>> from gofast.tools.datautils import replace_data
    >>> X, y = np.random.randn ( 7, 2 ), np.arange(7)
    >>> X.shape, y.shape 
    ((7, 2), (7,))
    >>> X_new, y_new = replace_data (X, y, n=10 )
    >>> X_new.shape , y_new.shape
    ((70, 2), (70,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> replace_data(X, n=2, axis=0)
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> replace_data(df, n=1, include_original=True, reset_index=True)
       A  B
    0  1  3
    1  2  4
    2  1  3
    3  2  4
    """
    def concat_data(ar) -> Union[np.ndarray, pd.DataFrame]:
        repeated_data = [ar] * (n + 1) if include_original else [ar] * n
        
        if random_sample:
            random_indices = np.random.choice(
                ar.shape[0], size=ar.shape[0], replace=True)
            repeated_data = [ar[random_indices] for _ in repeated_data]

        concatenated = pd.concat(repeated_data, axis=axis) if isinstance(
            ar, pd.DataFrame) else np.concatenate(repeated_data, axis=axis)
        
        if shuffle:
            shuffled_indices = np.random.permutation(concatenated.shape[0])
            concatenated = concatenated[shuffled_indices] if isinstance(
                ar, pd.DataFrame) else concatenated.iloc[shuffled_indices]

        if reset_index and isinstance(concatenated, pd.DataFrame):
            concatenated.reset_index(drop=True, inplace=True)
        
        return concatenated

    X = np.array(X) if not isinstance(X, (np.ndarray, pd.DataFrame)) else X
    y = np.array(y) if y is not None and not isinstance(y, (np.ndarray, pd.Series)) else y

    if y is not None:
        return concat_data(X), concat_data(y)
    return concat_data(X)

@isdf 
@validate_params ({ 
    "long_df": ['array-like'], 
    'index_columns': ['array-like', str, None], 
    'pivot_column': [str], 
    'value_column': [str], 
    'aggfunc': [StrOptions({'first', })], 
    'rename_columns': [list], 
    'rename-dict': [dict], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})]
    }
 )
def long_to_wide(
    long_df,
    index_columns=None,
    pivot_column='year',
    value_column='subsidence',
    aggfunc='first',
    sep='_', 
    name_prefix=None, 
    new_columns=None, 
    error ='warn', 
    exclude_value_from_name=False, 
    savefile=None, 
):
    """
    Convert a DataFrame from long to wide format by pivoting.

    This function transforms a DataFrame from long format to wide
    format by pivoting the DataFrame based on specified index columns,
    pivot column, and value column. The resulting DataFrame will have
    one row per unique combination of `index_columns` and columns for
    each unique value in `pivot_column`.

    Parameters
    ----------
    long_df : pandas.DataFrame
        The input DataFrame in long format containing the data to be
        pivoted.

    index_columns : list of str, optional
        List of column names to use as the index for the pivot
        operation. If `None`, defaults to ``['longitude', 'latitude']``.

    pivot_column : str, default ``'year'``
        The name of the column whose values will be used as new column
        names in the pivoted DataFrame.

    value_column : str, default ``'subsidence'``
        The name of the column whose values will fill the cells of the
        pivoted DataFrame.

    aggfunc : str or callable, default ``'first'``
        The aggregation function to apply if there are duplicate
        entries for the same index and pivot column values.
        
    sep : str, default='_'
        The string used to separate the value of `value_column` and
        `pivot_column` in the column names of the resulting wide-format
        DataFrame.
    
        For example, if `value_column='subsidence'` and `pivot_column='year'`,
        setting `separator='[]'` results in column names like
        `subsidence[2020]`, `subsidence[2021]`.
    
        This parameter allows customizing the naming convention for
        the wide-format DataFrame, which may be useful for specific
        analysis or presentation needs.
        
    name_prefix : str, optional
        If provided, this value will replace `value_column` in the
        column names (e.g., ``name_prefix_2020`` or ``name_prefix[2020]``).

    new_columns : list of str, optional
        If provided, this list will replace the column names of the
        resulting wide-format DataFrame. The length of the list must
        match the number of columns in the resulting DataFrame,
        otherwise a `ValueError` is raised.
        
    error : {'raise', 'warn', 'ignore'}, default='warn'
        Defines the behavior when the provided `new_columns` does not match
        the number of columns in the resulting `wide_df`.
    
        - `'raise'`: Raises a `ValueError` if the length of `new_columns`
          does not match the number of columns in `wide_df`.
        - `'warn'`: Issues a warning and skips renaming if the lengths do
          not match.
        - `'ignore'`: Silently ignores the mismatch and skips renaming.

    exclude_value_from_name : bool, default ``False``
        If True, the `value_column` name is excluded from the resulting
        column names, leaving only the `pivot_column` values as column
        names (e.g., ``2020``, ``2021``). When True, the `separator`
        parameter is ignored.
        
    savefile : str, optional
        If provided, the resulting wide-format DataFrame will be saved
        to this file path in CSV format.

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per combination of
        `index_columns` and columns for each unique value in
        `pivot_column`.

    Notes
    -----
    This function is useful for transforming data from long to wide
    format, which is often required for certain types of analysis or
    visualization. The pivot operation is performed using
    :func:`pandas.pivot_table` [1]_.

    If there are multiple entries for the same index and pivot column
    values, the specified `aggfunc` is applied to aggregate them. The
    aggregation is defined as:

    .. math::

        \text{Value}_{i,j} = \text{aggfunc}(\{ v_k \mid
        \text{index}_k = i, \text{pivot}_k = j \})

    where :math:`v_k` are the values in `value_column`, :math:`i` and
    :math:`j` represent the unique values in `index_columns` and
    `pivot_column`, respectively.

    Examples
    --------
    >>> from gofast.tools.datautils import pivot_long_to_wide
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'longitude': [1, 1, 2, 2],
    ...     'latitude': [3, 3, 4, 4],
    ...     'year': [2020, 2021, 2020, 2021],
    ...     'subsidence': [0.1, 0.2, 0.15, 0.25]
    ... })
    >>> wide_df = pivot_long_to_wide(data)
    >>> print(wide_df)
       longitude  latitude  subsidence_2020  subsidence_2021
    0          1         3             0.10             0.20
    1          2         4             0.15             0.25

    See Also
    --------
    pandas.pivot_table : Create a spreadsheet-style pivot table as a DataFrame.

    References
    ----------
    .. [1] Wes McKinney. "pandas: a foundational Python library for
       data analysis and statistics." Python for High Performance and
       Scientific Computing (2011): 1-9.

    """
    # Set default index columns if not provided
    if index_columns is None:
        index_columns = ['longitude', 'latitude']

    # Check that required columns exist in DataFrame
    required_columns = index_columns + [pivot_column, value_column]
    missing_columns = set(required_columns) - set(long_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    # Pivot the DataFrame
    wide_df = long_df.pivot_table(
        index=index_columns,
        columns=pivot_column,
        values=value_column,
        aggfunc=aggfunc
    )

    # Flatten the multi-level columns if necessary
    if isinstance(wide_df.columns, pd.MultiIndex):
        col_names = wide_df.columns.get_level_values(1)
    else:
        col_names = wide_df.columns

    if exclude_value_from_name:
        wide_df.columns = [f'{col}' for col in col_names]
    else:
        name_to_use = name_prefix if name_prefix is not None else value_column
        if sep =='[]':
            wide_df.columns = [
                f'{name_to_use}[{col}]' for col in col_names
            ]
        else:
            wide_df.columns = [
            f'{name_to_use}{sep}{col}' for col in wide_df.columns
            ]
 
    # Apply column renaming if provided
    if new_columns is not None:
        new_columns = is_iterable (
            new_columns, exclude_string=True, transform=True ) 
        err_msg =( 
            f"The length of rename_columns ({len(new_columns)}) "
            "does not match the number of columns in the DataFrame"
            f" ({len(wide_df.columns)})."
            )
        if len(new_columns) != len(wide_df.columns):
            if error =="warn": 
                warnings.warn(err_msg) 
            elif error =='raise': 
                raise ValueError(err_msg)  
            # ignore and pass 
        else: 
            wide_df.columns = new_columns
        
    # Reset the index to convert back to a normal DataFrame
    wide_df = wide_df.reset_index()

    if savefile:
        wide_df.to_csv(savefile, index=False)

    return wide_df

@isdf
@validate_params ({ 
    "wide_df": ['array-like'], 
    'id_vars': ['array-like', str, None], 
    'value_vars': ['array-like', str, None], 
    'value_name': [str], 
    'var_name': [str], 
    'rename_columns': [list], 
    'rename-dict': [dict], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})]
    }
 )
def wide_to_long(
    wide_df, 
    id_vars=None, 
    value_vars=None,
    value_name='value', 
    var_name='variable', 
    rename_columns=None, 
    rename_dict=None,
    error='raise',
    **kwargs
):
    """
    Convert a wide-format DataFrame to a long-format DataFrame.

    Parameters
    ----------
    wide_df : pandas.DataFrame
        The input DataFrame in wide format.

    id_vars : list of str, optional
        Column names to use as identifier variables (columns to keep as is).
        If None, all columns not specified in `value_vars` will be used.

    value_vars : list of str, optional
        Column names to unpivot. If None, all columns not in `id_vars` will be used.

    value_name : str, default='value'
        Name of the column that will contain the values from the wide DataFrame.

    var_name : str, default='variable'
        Name of the column that will contain the variable names from the
        wide DataFrame (e.g., the wide-format column headers).

    rename_columns : list of str, optional
        If provided, this list will replace the column names of the resulting
        DataFrame. The length of `rename_columns` must match the number of 
        resulting columns.

    rename_dict : dict, optional
        A dictionary mapping existing column names to new names. This allows
        selective renaming without needing to specify all column names.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Defines the behavior when the provided `rename_columns` or keys in
        `rename_dict` do not match the existing columns:
        - `'raise'`: Raise a `ValueError` if there is a mismatch.
        - `'warn'`: Issue a warning and skip renaming for mismatched entries.
        - `'ignore'`: Silently ignore mismatches and proceed.

    **kwargs
        Additional keyword arguments to pass to `pd.melt`, such as `col_level`.

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with one row per unique combination of
        `id_vars` and the original wide-format columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.datautils import wide_to_long 
    >>> wide_df = pd.DataFrame({
    ...     'id': [1, 2],
    ...     'longitude': [10, 20],
    ...     'latitude': [30, 40],
    ...     '2015': [0.1, 0.15],
    ...     '2016': [0.2, 0.25]
    ... })
    >>> long_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year'
    ... )
    >>> print(long_df)
       id  longitude  latitude  year  subsidence
    0   1         10        30  2015        0.10
    1   2         20        40  2015        0.15
    2   1         10        30  2016        0.20
    3   2         20        40  2016        0.25

    >>> # Using rename_columns
    >>> renamed_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year',
    ...     rename_columns=['ID', 'Lon', 'Lat', 'Year', 'Subsidence']
    ... )
    >>> print(renamed_df)
       ID  Lon  Lat  Year  Subsidence
    0   1   10   30  2015        0.10
    1   2   20   40  2015        0.15
    2   1   10   30  2016        0.20
    3   2   20   40  2016        0.25

    >>> # Using rename_dict
    >>> renamed_dict_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year',
    ...     rename_dict={'id': 'ID', 'longitude': 'Lon', 'latitude': 'Lat'}
    ... )
    >>> print(renamed_dict_df)
       ID  Lon  Lat  year  subsidence
    0   1   10   30  2015        0.10
    1   2   20   40  2015        0.15
    2   1   10   30  2016        0.20
    3   2   20   40  2016        0.25
    """
    # Input Validation
    if not isinstance(wide_df, pd.DataFrame):
        raise TypeError(
            "wide_df must be a pandas DataFrame,"
            f" got {type(wide_df)} instead.")
    
    if id_vars is not None:
        id_vars= is_iterable(id_vars, exclude_string= True, transform =True )
        missing_id_vars = set(id_vars) - set(wide_df.columns)
        if missing_id_vars:
            raise ValueError(
                "The following id_vars are not in"
                f" the DataFrame columns: {missing_id_vars}")
    
    if value_vars is not None:
        value_vars= is_iterable(value_vars, exclude_string= True, 
                                transform =True )
        missing_value_vars = set(value_vars) - set(wide_df.columns)
        if missing_value_vars:
            raise ValueError(
                f"The following value_vars are not"
                f" in the DataFrame columns: {missing_value_vars}")
    
    if rename_columns is not None and not isinstance(rename_columns, list):
        raise TypeError(
            "rename_columns must be a list of strings,"
            f" got {type(rename_columns)} instead.")
    
    if rename_dict is not None and not isinstance(rename_dict, dict):
        raise TypeError(
            "rename_dict must be a dictionary,"
            f" got {type(rename_dict)} instead.")
    
    # Determine id_vars and value_vars if not provided
    if id_vars is None:
        if value_vars is not None:
            id_vars = [col for col in wide_df.columns if col not in value_vars]
        else:
            # If neither id_vars nor value_vars are provided, melt all columns except one
            if wide_df.shape[1] < 2:
                raise ValueError(
                    "wide_df must have at least two columns to perform melt.")
            id_vars = [wide_df.columns[0]]
            value_vars = list(wide_df.columns[1:])
    else:
        if value_vars is None:
            value_vars = [col for col in wide_df.columns if col not in id_vars]
        elif not value_vars:
            raise ValueError("value_vars cannot be an empty list.")

    # Perform the melt operation
    try:
        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
            **kwargs
        )
    except Exception as e:
        raise ValueError(f"Error during melting the DataFrame: {e}")

    # Apply renaming if rename_columns or rename_dict is provided
    if rename_columns is not None:
        if len(rename_columns) != len(long_df.columns):
            err_msg = (
                f"The length of rename_columns ({len(rename_columns)}) "
                "does not match the number of columns in"
                f" the resulting DataFrame ({len(long_df.columns)})."
            )
            if error == 'warn':
                warnings.warn(err_msg)
            elif error == 'raise':
                raise ValueError(err_msg)
            # 'ignore' will silently skip renaming
        else:
            long_df.columns = rename_columns

    if rename_dict is not None:
        # Check if keys in rename_dict exist in the DataFrame
        existing_keys = set(rename_dict.keys()).intersection(long_df.columns)
        if not existing_keys and error == 'raise':
            raise ValueError(
                "None of the keys in rename_dict match the DataFrame columns.")
        if existing_keys != set(rename_dict.keys()):
            missing_keys = set(rename_dict.keys()) - existing_keys
            err_msg = (
                "The following keys in rename_dict do not match"
                f" any DataFrame columns and will be skipped: {missing_keys}"
            )
            if error == 'warn':
                warnings.warn(err_msg)
            elif error == 'raise':
                raise ValueError(err_msg)
            # 'ignore' will silently skip renaming these keys
        # Perform the renaming
        long_df = long_df.rename(
            columns={k: v for k, v in rename_dict.items() if k in long_df.columns})

    return long_df

@isdf 
@is_data_readable
def repeat_feature_accross(
    data: DataFrame,
    date_col: str = 'date',
    start_date: Union[int, pd.Timestamp] = None,
    end_date: Union[int, pd.Timestamp] = None,
    n_times: int = None,
    custom_dates: List[Union[int, pd.Timestamp]] = None,
    drop_existing_date: bool = True,
    sort: bool = False,
    inplace: bool = False
) -> DataFrame:
    """
    Repeat static feature across multiple years or specified dates.

    This function duplicates each row in the input DataFrame across a range of
    years, specific dates, or for a defined number of repetitions (`n_times`). 
    It is designed to handle various scenarios, ensuring flexibility and 
    robustness.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing static data to be repeated across dates.
    
    date_col : str, default='date'
        The name of the date column in the resulting DataFrame.
    
    start_date : int or pd.Timestamp, optional
        The starting date for duplication. Must be specified if 
        `end_date` is provided.
    
    end_date : int or pd.Timestamp, optional
        The ending date for duplication. Must be specified if 
        `start_date` is provided.
    
    n_times : int, optional
        The number of times to repeat the data. Overrides `start_date` 
        and `end_date` if provided.
    
    custom_dates : list of int or pd.Timestamp, optional
        A custom list of dates to duplicate the data across. Overrides 
        `start_date`, `end_date`, and `n_times` if provided.
    
    drop_existing_date : bool, default=True
        If `True`, drops the existing `date_col` in `data` before duplication
        to avoid conflicts.
    
    sort : bool, default=False
        If `True`, sorts the resulting DataFrame by the `date_col`.
    
    inplace : bool, default=False
        If `True`, modifies the input DataFrame `data` in place and returns it.
        If `False`, returns a new DataFrame with the duplicated data.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with the static data repeated across the specified dates. The
        resulting DataFrame includes the `date_col` indicating the date for each
        duplicated entry.

    Raises
    ------
    ValueError
        - If neither `custom_dates`, (`start_date` and `end_date`), nor 
          `n_times` is provided.
        - If `start_date` is provided without `end_date`, or vice versa.
        - If `n_times` is not a positive integer.
        - If `custom_dates` is not a list of integers or pandas Timestamps.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> from gofast.tools.datautils import repeat_feature_accross
    >>> data = {
    ...     'longitude': [113.291328, 113.291847, 113.291847],
    ...     'latitude': [22.862476, 22.865587, 22.865068],
    ...     'geology': ['Triassic', 'Carboniferous', 'Tertiary']
    ... }
    >>> data = pd.DataFrame(data)
    >>> repeated_df = repeat_feature_accross(
    ...     data, 
    ...     date_col='year', 
    ...     start_date=2015, 
    ...     end_date=2022
    ... )
    >>> print(repeated_df)
         longitude   latitude        geology  year
    0   113.291328  22.862476       Triassic  2015
    1   113.291328  22.862476       Triassic  2016
    2   113.291328  22.862476       Triassic  2017
    3   113.291328  22.862476       Triassic  2018
    4   113.291328  22.862476       Triassic  2019
    5   113.291328  22.862476       Triassic  2020
    ...
    19  113.291847  22.865068       Tertiary  2018
    20  113.291847  22.865068       Tertiary  2019
    21  113.291847  22.865068       Tertiary  2020
    22  113.291847  22.865068       Tertiary  2021
    23  113.291847  22.865068       Tertiary  2022
    
    >>> # Using n_times
    >>> repeated_df = repeat_feature_accross(
    ...     data, 
    ...     date_col='year', 
    ...     n_times=3
    ... )
    >>> print(repeated_df)
        longitude   latitude        geology  year
    0  113.291328  22.862476       Triassic     1
    1  113.291847  22.865587  Carboniferous     1
    2  113.291847  22.865068       Tertiary     1
    3  113.291328  22.862476       Triassic     2
    4  113.291847  22.865587  Carboniferous     2
    5  113.291847  22.865068       Tertiary     2
    6  113.291328  22.862476       Triassic     3
    7  113.291847  22.865587  Carboniferous     3
    8  113.291847  22.865068       Tertiary     3
    """
    # Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Expected 'data' to be a pandas DataFrame, got {type(data)} instead."
        )
    
    # Determine the list of dates/years
    if custom_dates is not None:
        if (not isinstance(custom_dates, list) or 
            not all(isinstance(
                date, (int, pd.Timestamp)) for date in custom_dates)):
            raise ValueError(
                "'custom_dates' must be a list of integers or pandas Timestamps."
            )
        dates = sorted(set(custom_dates))
    elif n_times is not None:
        n_times = validate_positive_integer(
            n_times, "n_times", msg= "'n_times' must be a positive integer."
            )
        if start_date is None and end_date is None:
            # Default to 1 to n_times
            dates = list(range(1, n_times + 1))
        elif start_date is not None and end_date is not None:
            if not isinstance(start_date, (int, pd.Timestamp)) or not isinstance(
                    end_date, (int, pd.Timestamp)):
                raise ValueError(
                    "'start_date' and 'end_date' must be integers or pandas Timestamps."
                )
            if isinstance(start_date, int) and isinstance(end_date, int):
                if start_date > end_date:
                    raise ValueError(
                        "'start_date' must be less than or equal to 'end_date'."
                    )
                dates = list(range(start_date, end_date + 1))
            elif isinstance(start_date, pd.Timestamp) and isinstance(
                    end_date, pd.Timestamp):
                if start_date > end_date:
                    raise ValueError(
                        "'start_date' must be earlier than or equal to 'end_date'."
                    )
                dates = pd.date_range(start=start_date, end=end_date, freq='Y').tolist()
            else:
                raise ValueError(
                    "'start_date' and 'end_date' must both"
                    " be integers or both be pandas Timestamps."
                )
        else:
            raise ValueError(
                "Both 'start_date' and 'end_date' must be provided together."
            )
    elif start_date is not None and end_date is not None:
        if not isinstance(start_date, (int, pd.Timestamp)) or not isinstance(
                end_date, (int, pd.Timestamp)):
            raise ValueError(
                "'start_date' and 'end_date' must be integers or pandas Timestamps.")
        if isinstance(start_date, int) and isinstance(end_date, int):
            if start_date > end_date:
                raise ValueError(
                    "'start_date' must be less than or equal to 'end_date'.")
            dates = list(range(start_date, end_date + 1))
        elif isinstance(
                start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
            if start_date > end_date:
                raise ValueError(
                    "'start_date' must be earlier than or equal to 'end_date'.")
            dates = pd.date_range(
                start=start_date, end=end_date, freq='Y').tolist()
        else:
            raise ValueError(
                "'start_date' and 'end_date' must both"
                " be integers or both be pandas Timestamps."
            )
    else:
        raise ValueError(
            "Must provide either 'custom_dates', "
            "('start_date' and 'end_date'), or 'n_times'."
        )
    
    # Validate date_col type consistency
    if isinstance(dates[0], pd.Timestamp):
        if not pd.api.types.is_datetime64_any_dtype(
                data.get(date_col, pd.Series(dtype='object'))):
            pass  # Allow creation of date_col as datetime
    elif isinstance(dates[0], int):
        if not pd.api.types.is_integer_dtype(
                data.get(date_col, pd.Series(dtype='object'))):
            pass  # Allow creation of date_col as integer
    
    # Handle existing date column
    if drop_existing_date and date_col in data.columns:
        data = data.drop(columns=[date_col])
    
    # Create a DataFrame with the dates to merge
    dates_df = pd.DataFrame({date_col: dates})
    
    # Perform cross join to duplicate rows across dates
    df_repeated = data.merge(dates_df, how='cross')
    
    # Sort if required
    if sort:
        df_repeated = df_repeated.sort_values(by=date_col).reset_index(drop=True)
    
    if inplace:
        data.drop(columns=data.columns, inplace=True)
        for col in df_repeated.columns:
            data[col] = df_repeated[col]
        return data
    else:
        return df_repeated.reset_index(drop=True)

def merge_datasets(
    *data, 
    on=None, 
    how='inner', 
    fill_missing=False, 
    fill_value=None, 
    keep_duplicates=False, 
    suffixes=('_x', '_y')
):
    """
    Merge multiple datasets into a single DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Variable-length arguments of DataFrames to be merged.

    on : list of str or None, default None
        The list of columns to join on. If None, the intersection of
        columns across datasets is used.

    how : {'inner', 'outer', 'left', 'right'}, default 'inner'
        Type of merge to be performed:
        - 'inner': Only include rows with matching keys in all datasets.
        - 'outer': Include all rows from all datasets, filling missing
          values with NaN.
        - 'left': Include all rows from the first dataset and matching
          rows from others.
        - 'right': Include all rows from the last dataset and matching
          rows from others.

    fill_missing : bool, default False
        If True, fills missing values with a default value.

    fill_value : any, default None
        The value to use when `fill_missing` is True. If None, numeric
        columns are filled with 0, and non-numeric columns are filled
        with an empty string.

    keep_duplicates : bool, default False
        If True, keeps duplicate rows across datasets after merging. If
        False, removes duplicates from the merged DataFrame.

    suffixes : tuple of (str, str), default ('_x', '_y')
        Suffixes to apply to overlapping column names when merging.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing all the data.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.datautils import merge_datasets
    >>> df1 = pd.DataFrame({'longitude': [1, 2], 'latitude': [3, 4],
    ...                     'year': [2020, 2021], 'value1': [10, 20]})
    >>> df2 = pd.DataFrame({'longitude': [1, 2], 'latitude': [3, 4],
    ...                     'year': [2020, 2021], 'value2': [100, 200]})
    >>> merged = merge_datasets(df1, df2, on=['longitude', 'latitude',
    ...                                       'year'], how='inner')
    >>> print(merged)
       longitude  latitude  year  value1  value2
    0          1         3  2020      10     100
    1          2         4  2021      20     200
    """
    [ is_frame (d, df_only=True, raise_exception=True, objname='Dataset')
            for d in data 
    ]
    if len(data) < 2:
        raise ValueError(
            "At least two DataFrames are required for merging."
        )
    # Ensure all arguments are DataFrames
    for df in data:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}.")

    if on is None:
        # Use the intersection of all datasets' columns for merging
        common_columns = set(data[0].columns)
        for df in data[1:]:
            common_columns.intersection_update(df.columns)
        on = list(common_columns)
    
    if not on:
        raise ValueError(
            "No common columns found for merging. Specify the 'on' parameter."
        )

    # Perform iterative merging
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, on=on, how=how, suffixes=suffixes), data)

    # Fill missing values if required
    if fill_missing:
        if fill_value is None:
            for column in merged_df:
                if merged_df[column].dtype.kind in 'biufc':  # Numeric types
                    merged_df[column].fillna(0, inplace=True)
                else:  # Categorical or string types
                    merged_df[column].fillna('', inplace=True)
        else:
            merged_df.fillna(fill_value, inplace=True)

    # Remove duplicates if specified
    if not keep_duplicates:
        merged_df.drop_duplicates(inplace=True)

    return merged_df

@isdf 
def swap_ic(
    data, 
    sort: bool = False, 
    ascending: bool = True, 
    inplace: bool = False, 
    reset_index: bool = False, 
    dropna: bool = False, 
    fillna: bool = None, 
    axis: int = 0, 
    order: list = None,
    **kwargs
    ):
    """
    Align the index and columns of a DataFrame so that they follow the 
    same order.

    This function ensures that if the values in the index and columns 
    are the same, the DataFrame will align its index and columns in 
    the same order. Optionally, the index and columns can be sorted, reset,
    and cleaned with additional parameters for flexibility.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame whose index and columns need to be aligned.

    sort : bool, optional, default=False
        Whether to sort the index and columns in ascending order. If `True`, both 
        index and columns will be sorted in ascending order.

    ascending : bool, optional, default=True
        If `sort=True`, this specifies the sorting order. If `True`, the 
        index and columns will be sorted in ascending order. Otherwise, 
        descending order will be applied.

    inplace : bool, optional, default=False
        If `True`, modifies the DataFrame in place. If `False`, 
        returns a new DataFrame.

    reset_index : bool, optional, default=False
        If `True`, resets the index after aligning the index and columns.

    dropna : bool, optional, default=False
        If `True`, rows or columns with NaN values will be dropped.

    fillna : scalar or dict, optional, default=None
        Value to replace NaNs with. If `None`, no filling is performed. 
        If a scalar, all NaNs will be replaced with that value. If a dict, 
        it should provide mappings for index and columns.

    axis : int, optional, default=0
        Axis along which to perform alignment. `0` for index, `1` for columns.
        
    order : list, optional, default=None
        Custom order for the index and columns. If specified, the index
        and columns will be ordered according to the provided list. If the
        order is not present in either index or columns, it will be ignored. 
        If `None`, no custom order is applied.

    kwargs : additional keyword arguments
        Any additional arguments that might be passed to the DataFrame operations.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame (or the original, if `inplace=True`) with aligned index 
        and columns, with optional sorting, resetting, and cleaning applied.
    
    Raises
    ------
    ValueError
        If the index and columns are not the same, the function will 
        raise an error  or allow custom handling if required.
    
    Examples
    --------
    Example of using the function without sorting:

    >>> from gofast.tools.datautils import swap_ic 
    >>> data = pd.DataFrame({
    >>>     'A': [1, 2, 3],
    >>>     'B': [4, 5, 6],
    >>>     'C': [7, 8, 9]
    >>> }, index=['B', 'A', 'C'])
    >>> swap_ic(data, sort=False)
    >>> swap_ic(data, sort=True, ascending=True)

    Example of using the function with sorting and filling NaNs:

    >>> swap_ic(data, sort=True, fillna=0)
    
    Example of using the function with custom order:

    >>> swap_ic(data, order=['B', 'A', 'C'])
    >>> swap_ic(data, order=['C', 'A', 'B'])
    
    """
    # Validate that index and columns have the same elements
    if not set(data.columns).issubset(data.index) or not set(
            data.index).issubset(data.columns):
        raise ValueError("Index and columns must contain the same values.")
    
    # Optionally, apply custom order to both index and columns
    if order:
        # Ensure custom_order is a valid subset of index and columns
        if not set(order).issubset(data.columns) or not set(order).issubset(data.index):
            raise ValueError(
                "Custom order contains values not present in both index and columns.")
        
        # Apply custom order to index and columns
        data = data.loc[order, order]
        
    # Align index and columns by ensuring they are in the same order
    aligned_data = data.loc[data.columns, data.columns]

    # Sort the index and columns if requested
    if sort:
        aligned_data = aligned_data.sort_index(ascending=ascending, axis=0)
        aligned_data = aligned_data.sort_index(ascending=ascending, axis=1)

    # Reset index if requested
    if reset_index:
        aligned_data = aligned_data.reset_index(drop=True)

    # Drop NaN values if requested
    if dropna:
        aligned_data = aligned_data.dropna(axis=axis, how='any')

    # Fill NaN values if requested
    if fillna is not None:
        aligned_data = aligned_data.fillna(fillna)

    # Apply changes in place if requested
    if inplace:
        data[:] = aligned_data
        return None  # None is returned if inplace=True
    else:
        return aligned_data
    

