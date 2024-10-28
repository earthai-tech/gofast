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
import warnings 
from typing import Any, List, Union, Dict, Optional, Set  

import scipy 
import numpy as np 
import pandas as pd 

from ..api.types import _F, ArrayLike, NDArray, DataFrame  
from .coreutils import ( 
    smart_format, is_iterable, _assert_all_types, exist_features, 
    assert_ratio, _isin, _validate_name_in, sanitize_frame_cols, 
    to_numeric_dtypes 
    )

__all__= [
    'cleaner',
    'get_xy_coordinates',
    'nan_to_na',
    'pair_data',
    'process_and_extract_data',
    'random_sampling',
    'random_selector',
    'read_from_excelsheets',
    'read_worksheets',
    'resample_data'
    ]


def nan_to_na(
    data: DataFrame, 
    cat_missing_value: Optional[Union[Any,  float]]=pd.NA, 
    nan_spec:Optional[Union[Any, float]]=np.nan
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
    >>> from gofast.tools.coreutils import nan_to_na
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1.0, 2.0, np.nan], 'B': ['x', np.nan, 'z']})
    >>> df['B'] = df['B'].astype('category')
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
    if not isinstance (data, (pd.Series, pd.DataFrame)): 
        raise ValueError("Input must be a pandas DataFrame or Series."
                         f" Got {type(data).__name__!r} instead.")
        
    def has_nan_values(series, nan_spec):
        """Check if nan_spec exists in the series."""
        return series.isin([nan_spec]).any()
    
    if isinstance(data, pd.Series):
        if has_nan_values(data, nan_spec):
            if pd.api.types.is_categorical_dtype(data):
                return data.replace({nan_spec: cat_missing_value})
        return data
    
    elif isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        for column in df_copy.columns:
            if has_nan_values(df_copy[column], nan_spec):
                if pd.api.types.is_categorical_dtype(df_copy[column]):
                    df_copy[column] = df_copy[column].replace({nan_spec: cat_missing_value})
        return df_copy

def resample_data(*data: Any, samples: Union[int, float, str] = 1,
                  replace: bool = False, random_state: int = None,
                  shuffle: bool = True) -> List[Any]:
    """
    Resample multiple data structures (arrays, sparse matrices, Series, DataFrames)
    based on specified sample size or ratio.

    Parameters
    ----------
    *data_structures : Any
        Variable number of array-like, sparse matrix, pandas Series, or DataFrame
        objects to be resampled.
    samples : int, float, or str, optional
        Number of items to sample or a ratio of items to sample (if < 1 or
        expressed as a percentage string, e.g., "50%"). Defaults to 1, meaning
        no resampling unless specified.
    replace : bool, default=False
        Whether sampling of the same row more than once is allowed.
    random_state : int, optional
        Seed for the random number generator for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle data before sampling.

    Returns
    -------
    List[Any]
        A list of resampled data structures, matching the input order. Each
        structure is resampled according to the `samples` parameter.

    Examples
    --------
    >>> from gofast.tools.coreutils import resample_data
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris(as_frame=True)
    >>> data, target = iris.data, iris.target
    >>> resampled_data, resampled_target = resample_data(data, target, samples=0.5,
    ...                                                  random_state=42)
    >>> print(resampled_data.shape, resampled_target.shape)
    """
    resampled_structures = []
    
    for d in data:
        # Correct way to access the shape of the sparse matrix 
        # encapsulated in a numpy array
        try: 
            if d.dtype == object and scipy.sparse.issparse(d.item()):
                d = d.item()  # Access the sparse matrix
            # Now you can safely access the number of rows
        except:
            # Fallback for regular numpy arrays/data or 
            # directly accessible sparse matrices
           pass 
        n_samples = _determine_sample_size(d, samples, is_percent="%" in str(samples))
        sampled_d = _perform_sampling(d, n_samples, replace, random_state, shuffle)
        resampled_structures.append(sampled_d)

    return resampled_structures

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
    *d: Union[pd.DataFrame, List[pd.DataFrame]],  
    on: Union[str, List[str]] = None, 
    parse_on: bool = False, 
    mode: str = 'strict', 
    coerce: bool = False, 
    force: bool = False, 
    decimals: int = 7, 
    raise_warn: bool = True 
) -> pd.DataFrame:
    """ Find indentical object in all data and concatenate them using merge 
     intersection (`cross`) strategy.
    
    Parameters 
    ---------- 
    d: List of DataFrames 
       List of pandas DataFrames 
    on: str, label or list 
       Column or index level names to join on. These must be found in 
       all DataFrames. If `on` is ``None`` and not merging on indexes then 
       a concatenation along columns axis is performed in all DataFrames. 
       Note that `on` works with `parse_on` if its argument is  a list of 
       columns names passed into single litteral string. For instance:: 
           
        on ='longitude latitude' --[parse_on=True]-> ['longitude' , 'latitude'] 
        
    parse_on: bool, default=False 
       Parse `on` arguments if given as string and return_iterable objects. 
       
    mode: str, default='strict' 
      Mode to the data. Can be ['soft'|'strict']. In ``strict`` mode, all the 
      data passed must be a DataFrame, otherwise an error raises. in ``soft``
      mode, ignore the non-DataFrame. Note that any other values should be 
      in ``strict`` mode. 
      
    coerce: bool, default=False 
       Truncate all DataFrame size to much the shorter one before performing 
       the ``merge``. 
        
    force: bool, default=False, 
       Force `on` items to be in the all DataFrames, This could be possible 
       at least, `on` items should be in one DataFrame. If missing in all 
       data, an error occurs.  
 
    decimals: int, default=5 
       Decimal is used for comparison between numeric labels in `on` columns 
       items. If set, it rounds values of `on` items in all data before 
       performing the merge. 
       
     raise_warn: bool, default=False 
        Warn user to concatenate data along column axis if `on` is ``None``. 

    Returns 
    --------
    data: DataFrames 
      A DataFrame of the merged objects.
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.coreutils import pair_data 
    >>> data = gf.make_erp (seed =42 , n_stations =12, as_frame =True ) 
    >>> table1 = gf.DCProfiling ().fit(data).summary()
    >>> table1 
           dipole   longitude  latitude  ...  shape  type       sfi
    line1      10  110.486111  26.05174  ...      C    EC  1.141844
    >>> data_no_xy = gf.make_ves ( seed=0 , as_frame =True) 
    >>> data_no_xy.head(2) 
        AB   MN  resistivity
    0  1.0  0.4   448.860148
    1  2.0  0.4   449.060335
    >>> data_xy = gf.make_ves ( seed =0 , as_frame =True , add_xy =True ) 
    >>> data_xy.head(2) 
        AB   MN  resistivity   longitude  latitude
    0  1.0  0.4   448.860148  109.332931  28.41193
    1  2.0  0.4   449.060335  109.332931  28.41193
    >>> table = gf.methods.VerticalSounding (
        xycoords = (110.486111,   26.05174)).fit(data_no_xy).summary() 
    >>> table.table_
             AB    MN   arrangememt  ... nareas   longitude  latitude
    area                             ...                             
    None  200.0  20.0  schlumberger  ...      1  110.486111  26.05174
    >>> pair_data (table1, table.table_,  ) 
           dipole   longitude  latitude  ...  nareas   longitude  latitude
    line1    10.0  110.486111  26.05174  ...     NaN         NaN       NaN
    None      NaN         NaN       NaN  ...     1.0  110.486111  26.05174
    >>> pair_data (table1, table.table_, on =['longitude', 'latitude'] ) 
    Empty DataFrame 
    >>> # comments: Empty dataframe appears because, decimal is too large 
    >>> # then it considers values longitude and latitude differents 
    >>> pair_data (table1, table.table_, on =['longitude', 'latitude'], decimals =5 ) 
        dipole  longitude  latitude  ...  max_depth  ohmic_area  nareas
    0      10  110.48611  26.05174  ...      109.0  690.063003       1
    >>> # Now is able to find existing dataframe with identical closer coordinates. 
    
    """
    from .validator import _is_numeric_dtype  

    if str(mode).lower()=='soft': 
        d = [ o for  o in d if hasattr (o, '__array__') and hasattr (o, 'columns') ]
    
    is_same = set ( [ hasattr (o, '__array__') 
                     and hasattr (o, 'columns') for o in d ] ) 
    
    if len(is_same)!=1 or not list (is_same) [0]: 
        types = [ type(o).__name__ for o in d ]
        raise TypeError (
            f"Expect DataFrame. Got {smart_format(types)}")
    
    same_len = [len(o) for o in d] 
    
    if len( set(same_len)) !=1 or not list(set(same_len))[0]: 
        if not coerce: 
            raise ValueError(
                f"Data must be a consistent size. Got {smart_format(same_len)}"
                " respectively. Set ``coerce=True`` to truncate the data"
                " to match the shorter data length.")
        # get the shorthest len 
        min_index = min (same_len) 
        d = [ o.iloc [:min_index, :  ]  for o in d ] 

    if on is None:
        if raise_warn: 
            warnings.warn("'twin_items' are missing in the data. A simple merge"
                          " along the columns axis should be performed.") 
        
        return pd.concat ( d, axis = 1 )
    
    # parse string 
    on= is_iterable(on, exclude_string= True , 
                    transform =True, parse_string= parse_on  
                    )
    
    feature_exist = [
        exist_features(o, on, error = 'ignore'
                       ) for o in d ]  

    if ( len( set (feature_exist) )!=1 
        or not list(set(feature_exist)) [0]
            ): 
        if not force: 
            raise ValueError(
                f"Unable to fit the data. Items {smart_format(on)} are"
                f" missing in the data columns. {smart_format(on)} must"
                 " include in the data columns. Please check your data.")

        # seek the value twin_items in the data and use if for all  
        dtems =[] ;  repl_twd= None 
        for o in d: 
            # select one valid data that contain the 
            # twin items
            try : 
                exist_features ( o, on ) 
            except : 
                pass 
            else:
                repl_twd = o [ on ]
                break 
            
        if repl_twd is None: 
            raise ValueError("To force data that have not consistent items,"
                             f" at least {smart_format(on)} items must"
                             " be included in one DataFrame.")
        # make add twin_data if 
        for o in d : 
            try : exist_features(o, on) 
            except : 
                a = o.copy()  
                a [ on ] = repl_twd 
            else : a = o.copy () 
            
            dtems.append(a)
        # reinitialize d
        d = dtems  
    
    # check whether values to merge are numerics 
    # if True, use decimals to round values to consider 
    # that identic 
    # round value if value before performed merges 
    # test single data with on 
    is_num = _is_numeric_dtype (d[0][on] ) 
    if is_num: 
        decimals = int (_assert_all_types(
            decimals, int, float, objname ='Decimals'))
        d_ = []
        for o in d : 
            a = o.copy()  
            a[on ] = np.around (o[ on ].values, decimals ) 
            d_.append (a )
        # not a numerick values so stop     
        d =d_

    # select both two  
    data = pd.merge (* d[:2], on= on ) 
    
    if len(d[2:]) !=0: 
        for ii, o in enumerate ( d[2:]) : 
            data = pd.merge ( *[data, o] , on = on, suffixes= (
                f"_x{ii+1}", f"_y{ii+1}")) 
            
    return data 

def random_sampling (
    d,  
    samples:int = None  , 
    replace:bool = False , 
    random_state:int = None, 
    shuffle=True, 
    ): 
    """ Sampling data. 
    
    Parameters 
    ----------
    d: {array-like, sparse matrix} of shape (n_samples, n_features)
      Data for sampling, where `n_samples` is the number of samples and
      `n_features` is the number of features.
    samples: int,optional 
       Ratio or number of items from axis to return. 
       Default = 1 if `samples` is ``None``.
       
    replace: bool, default=False
       Allow or disallow sampling of the same row more than once. 
       
    random_state: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    shuffle:bool, default=True 
       Shuffle the data before sampling 
      
    Returns 
    ----------
    d: {array-like, sparse matrix} of shape (n_samples, n_features)
    samples data based on the given samples. 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import random_sampling 
    >>> from gofast.datasets import load_hlogs 
    >>> data= load_hlogs().frame
    >>> random_sampling( data, samples = 7 ).shape 
    (7, 27)
    """

    n= None ; is_percent = False
    orig= copy.deepcopy(samples )
    if not hasattr(d, "__iter__"): 
        d = is_iterable(d, exclude_string= True, transform =True )
    
    if ( 
            samples is None 
            or str(samples) in ('1', '*')
            ): 
        samples =1. 
        
    if "%" in str(samples): 
        samples = samples.replace ("%", '')
        is_percent=True 
    # assert value for consistency. 
    try: 
        samples = float( samples)
    except: 
        raise TypeError("Wrong value for 'samples'. Expect an integer."
                        f" Got {type (orig).__name__!r}")

    if samples <=1 or is_percent: 
        samples  = assert_ratio(
            samples , bounds = (0, 1), exclude_value= 'use lower bound', 
            in_percent= True )
        
        n = int ( samples * ( d.shape[0] if scipy.sparse.issparse(d)
                 else len(d))) 
    else: 
        # data frame 
        n= int(samples)
        
    # reset samples and use number of samples instead    
    samples =None  
    # get the total length of d
    dlen = ( d.shape[0] if scipy.sparse.issparse(d) else len(d))
    # if number is greater than the length 
    # block to the length so to retrieve all 
    # value no matter the arrangement. 
    if n > dlen: 
        n = dlen
    if hasattr (d, 'columns') or hasattr (d, 'name'): 
        # data frame 
        return d.sample ( n= n , frac=samples , replace = replace ,
                         random_state = random_state  
                     ) if shuffle else d.iloc [ :n , ::] 
        
    np.random.seed ( random_state)
    if scipy.sparse.issparse(d) : 
        if scipy.sparse.isspmatrix_coo(d): 
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " should be performed in CSR matrix")
            d = d.tocsr() 
            
        return d [ np.random.choice(
            np.arange(d.shape[0]), n, replace=replace )] if shuffle else d [
                [ i for i in range (n)]]
                
        #d = d[idx ]
        
    # manage the data 
    if not hasattr(d, '__array__'): 
        d = np.array (d ) 
        
    idx = np.random.randint( len(d), size = n ) if shuffle else [ i for i in range(n)]
    if len(d.shape )==1: d =d[idx ]
    else: d = d[idx , :]
        
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

def read_worksheets(*data): 
    """ Read sheets and returns a list of DataFrames and sheet names. 
    
    Parameters 
    -----------
    data: list of str 
      A collection of excel sheets files. Read only `.xlsx` files. Any other 
      files raises an errors.  
    
    Return
    ------
    data, sheet_names: Tuple of DataFrames and sheet_names 
       A collection of DataFrame and sheets names. 
       
    Examples 
    -----------
    >>> import os 
    >>> from gofast.tools.coreutils import read_worksheets 
    >>> sheet_file= r'_F:\repositories\gofast\data\erp\sheets\gbalo.xlsx'
    >>> data, snames =  read_worksheets (sheet_file )
    >>> snames 
    ['l11', 'l10', 'l02'] 
    >>> data, snames =  read_worksheets (os.path.dirname (sheet_file))
    >>> snames 
    ['l11', 'l10', 'l02', 'l12', 'l13']
    
    """
    dtem = []
    data = [o for o in data if isinstance ( o, str )]
    
    for o in data: 
        if os.path.isdir (o): 
            dlist = os.listdir (o)
            # collect only the excell sheets 
            p = [ os.path.join(o, f) for f in dlist if f.endswith ('.xlsx') ]
            dtem .extend(p)
        elif os.path.isfile (o):
            _, ex = os.path.splitext( o)
            if ex == '.xlsx': 
                dtem.append(o)
            
    data = copy.deepcopy(dtem)
    # if no excel sheets is found return None 
    if len(data) ==0: 
        return None, None 
    
    # make d dict to collect data 
    ddict = dict() 
    regex = re.compile (r'[$& #@%^!]', flags=re.IGNORECASE)
    
    for d in data : 
        try: 
            ddict.update ( **pd.read_excel (d , sheet_name =None))
        except : pass 

    #collect stations names
    if len(ddict)==0 : 
        raise TypeError("Can'find the data to read.")

    sheet_names = list(map(
        lambda o: regex.sub('_', o).lower(), ddict.keys()))

    data = list(ddict.values ()) 

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
    >>> from gofast.tools.coreutils import process_and_extract_data
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
            raise ValueError("Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def random_selector (
        arr:ArrayLike, value: Union [float, ArrayLike], 
        seed: int = None, shuffle =False ): 
    """Randomly select the number of values in array. 
    
    Parameters
    ------------
    arr: ArrayLike 
       Array of values 
    value: float, arraylike 
        If ``float`` value is passed, it indicates the number of values to 
        select among the length of `arr`. If array (``value``) is passed, it
        should be self contain in the given ``arr`. However if ``string`` is 
        given and contain the ``%``, it calculates the ratio of 
        number to randomly select. 
    seed: int, Optional 
       Allow retrieving the identical value randomly selected in the given 
       array. 
       
    suffle: bool, False 
       If  ``True`` , shuffle the selected values. 
       
    Returns 
    --------
    arr: Array containing the selected values 
     
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import random_selector 
    >>> dat= np.arange (42 ) 
    >>> random_selector (dat , 7, seed = 42 ) 
    array([0, 1, 2, 3, 4, 5, 6])
    >>> random_selector ( dat, ( 23, 13 , 7))
    array([ 7, 13, 23])
    >>> random_selector ( dat , "7%", seed =42 )
    array([0, 1])
    >>> random_selector ( dat , "70%", seed =42 , shuffle =True )
    array([ 0,  5, 20, 25, 13,  7, 22, 10, 12, 27, 23, 21, 16,  3,  1, 17,  8,
            6,  4,  2, 19, 11, 18, 24, 14, 15,  9, 28, 26])
    """
    
    msg = "Non-numerical is not allowed. Got {!r}."
    
    if seed: 
        seed = _assert_all_types(seed , int, float, objname ='Seed')
        np.random.seed (seed ) 
       
    v = copy.deepcopy(value )
    
    if not is_iterable( value, exclude_string= True ):
        
        value = str(value )
        
        if '%' in  value: 
            try: 
               value = float( value.replace ('%', '')) /100 
            except : 
                raise TypeError(msg.format(v))
            # get the number 
            value *= len(arr )
                
        
        try : 
            value = int(value )
            
        except :
            raise TypeError (msg.format(v))
    
        if value > len(arr): 
            raise ValueError(f"Number {value} is out of the range."
                             f" Expect value less than {len(arr)}.")
            
        value = np.random.permutation(value ) 
        
    arr = np.array ( 
        is_iterable( arr, exclude_string=True, transform =True )) 
    
    arr = arr.ravel() if arr.ndim !=1 else arr 

    mask = _isin (arr, value , return_mask= True )
    arr = arr [mask ] 
    
    if shuffle : np.random.shuffle (arr )

    return arr

def cleaner(
    data: Union[DataFrame, NDArray],
    columns: List[str] = None,
    inplace: bool = False,
    labels: List[Union[int, str]] = None,
    func: _F = None,
    mode: str = 'clean',
    **kws
) -> Union[DataFrame, NDArray, None]:
    """ Sanitize data or columns by dropping specified labels 
    from rows or columns. 
    
    If data is not a pandas dataframe, should be converted to 
    dataframe and uses index to drop the labels. 
    
    Parameters 
    -----------
    data: pd.Dataframe or arraylike2D. 
       Dataframe pandas or Numpy two dimensional arrays. If 2D array is 
       passed, it should prior be converted to a daframe by default and 
       drop row index from index parameters 
       
    columns: single label or list-like
        Alternative to specifying axis (
            labels, axis=1 is equivalent to columns=labels).

    labels: single label or list-like
      Index or column labels to drop. A tuple will be used as a single 
      label and not treated as a list-like.

    func: _F, callable 
        Universal function used to clean the columns. If performs only when 
        `mode` is on ``clean`` option. 
        
    inplace: bool, default False
        If False, return a copy. Otherwise, do operation 
        inplace and return None.
       
    mode: str, default='clean' 
       Options or mode of operation to do on the data. It could 
       be ['clean'|'drop']. If ``drop``, it behaves like ``dataframe.drop`` 
       of pandas. 
       
    Returns
    --------
    DataFrame, array2D  or None
            DataFrame cleaned or without the removed index or column labels 
            or None if inplace=True or array is data is passed as an array. 
            
    """
    mode = _validate_name_in(mode , defaults =("drop", 'remove' ), 
                      expect_name ='drop')
    if not mode: 
        return sanitize_frame_cols(
            data, 
            inplace = inplace, 
            func = func 
            ) 
 
    objtype ='ar'
    if not hasattr (data , '__array__'): 
        data = np.array (data ) 
        
    if hasattr(data , "columns"): 
        objtype = "pd" 
    
    if objtype =='ar': 
        data = pd.DataFrame(data ) 
        # block inplace to False and 
        # return numpy ar 
        inplace = False 
    # if isinstance(columns , str): 
    #     columns = str2columns(columns ) 
    if columns is not None: 
        columns = is_iterable(
            columns, exclude_string=True ,
            parse_string= True, 
            transform =True )
        
    data = data.drop (labels = labels, 
                      columns = columns, 
                      inplace =inplace,  
                       **kws 
                       ) 
    # re-verify integrity 
    # for consistency
    data = to_numeric_dtypes(data )
    return np.array ( data ) if objtype =='ar' else data 

def get_xy_coordinates (d, as_frame = False, drop_xy = False, 
                        raise_exception = True, verbose=0 ): 
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
    >>> from gofast.tools.coreutils import get_xy_coordinates 
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
       
