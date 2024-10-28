# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Input/Output utilities module for managing file paths, directories, and
loading serialized data. These functions support automated creation of
directories and error-checked deserialization of data, streamlining file
management and data recovery processes.
"""
import os 
import re
import shutil 
import pickle 
import joblib 
import datetime 
import warnings 
import copy 
import csv 
import json 
import yaml
import h5py 
from zipfile import ZipFile
from pathlib import Path
from six.moves import urllib  
from typing import Any, Optional , Tuple , Union, List, Dict, Text 

import numpy as np 
import pandas as pd 

__all__ = [
    'cparser_manager',
    'cpath',
    'dummy_csv_translator',
    'fetch_json_data_from_url',
    'get_config_fname_from_varname',
    'is_in_if',
    'key_checker',
    'key_search',
    'load_serialized_data',
    'move_cfile',
    'parse_csv',
    'parse_json',
    'parse_md_data',
    'parse_yaml',
    'print_cmsg',
    'rename_files',
    'replace_data',
    'return_ctask',
    'sanitize_unicode_string',
    'save_job',
    'save_path',
    'serialize_data',
    'spath',
    'store_or_write_hdf5',
    'str2columns',
    'to_hdf5',
    'to_iterable',
    'zip_extractor'
 ]

def to_iterable(
    obj: Any,
    exclude_string: bool = False,
    transform: bool = False,
    parse_string: bool = False,
    flatten: bool = False,
    unique: bool = False,
    delimiter: str = r'[ ,;|\t\n]+'
) -> Union[bool, List[Any]]:
    """
    Determines if an object is iterable, with options to transform, parse,
    and modify the input for flexible iterable handling.

    Parameters
    ----------
    obj : Any
        Object to be evaluated or transformed into an iterable.
    exclude_string : bool, default=False
        Excludes strings from being considered as iterable objects.
    transform : bool, default=False
        Transforms `obj` into an iterable if it isn't already. Defaults to
        wrapping `obj` in a list.
    parse_string : bool, default=False
        If `obj` is a string, splits it into a list based on the specified
        `delimiter`. Requires `transform=True`.
    flatten : bool, default=False
        If `obj` is a nested iterable, flattens it into a single list.
    unique : bool, default=False
        Ensures unique elements in the output if `transform=True`.
    delimiter : str, default=r'[ ,;|\t\n]+'
        Regular expression pattern for splitting strings when `parse_string=True`.

    Returns
    -------
    bool or List[Any]
        Returns a boolean if `transform=False`, or an iterable if
        `transform=True`.

    Raises
    ------
    ValueError
        If `parse_string=True` without `transform=True`, or if `delimiter`
        is invalid.

    Notes
    -----
    - When `parse_string` is used, strings are split by `delimiter` to form a
      list of substrings.
    - `flatten` and `unique` apply only when `transform=True`.
    - Using `unique=True` ensures no duplicate values in the output.

    Examples
    --------
    >>> from gofast.tools.iotuils import to_iterable
    >>> to_iterable("word", exclude_string=True)
    False

    >>> to_iterable(123, transform=True)
    [123]

    >>> to_iterable("parse, this sentence", transform=True, parse_string=True)
    ['parse', 'this', 'sentence']

    >>> to_iterable([1, [2, 3], [4]], transform=True, flatten=True)
    [1, 2, 3, 4]

    >>> to_iterable("a,b,a,b", transform=True, parse_string=True, unique=True)
    ['a', 'b']
    """
    if parse_string and not transform:
        raise ValueError("Set 'transform=True' when using 'parse_string=True'.")

    # Check if object is iterable (excluding strings if specified)
    is_iterable = hasattr(obj, '__iter__') and not (exclude_string and isinstance(obj, str))

    # If transformation is not needed, return the boolean check
    if not transform:
        return is_iterable

    # If string parsing is enabled and obj is a string, split it using delimiter
    if isinstance(obj, str) and parse_string:
        obj = re.split(delimiter, obj.strip())

    # Wrap non-iterables into a list if they aren't iterable
    elif not is_iterable:
        obj = [obj]

    # Flatten nested iterables if flatten=True
    if flatten:
        obj = _flatten(obj)

    # Apply unique filtering if requested
    if unique:
        obj = list(dict.fromkeys(obj))  # Preserves order while ensuring uniqueness

    return obj


def _flatten(nested_list: Any) -> List[Any]:
    """ Helper function to recursively flatten a nested list structure. """
    flattened = []
    for element in nested_list:
        if isinstance(element, (list, tuple, set)):
            flattened.extend(_flatten(element))
        else:
            flattened.append(element)
    return flattened

def zip_extractor(
    zip_file ,
    samples ='*', 
    ftype=None,  
    savepath = None,
    pwd=None,  
): 
    """ Extract  ZIP archive objects. 
    
    Can extract all or a sample objects when the number of object is passed 
    to the parameter ``samples``. 
    

    Parameters 
    -----------
    zip_file: str
        Full Path to archive Zip file. 
    samples: int, str, default ='*'
       Number of data to retrieve from archive files. This is useful when 
       the archive file contains many data. ``*`` means extract all. 
    savepath: str, optional 
       Path to store the decompressed archived files.
    ftype: str, 
       Is the extension of a specific file to decompress. Indeed, if the 
       archived files contains many different data formats, specifying the 
       data type would retrieve this specific files from the whole 
       files archieved. 
    pwd: int, optional
      Password to pass if the zip file is encrypted.
      
    Return 
    --------
    objnames: list, 
     List of decompressed objects. 
     
    Examples 
    ----------
    >>> from gofast.tools.coreutils import zip_extractor 
    >>> zip_extractor ('gofast/datasets/data/edis/e.E.zip')
    
    """
    def raise_msg_when ( objn, ft): 
        """ Raise message when None file is detected when the type of 
        of file is supplied. Otherwise return the object collected 
        from this kind of data-types
        """
        objn = [ o for o  in objn if o.endswith (ft)]
        if len(objn)  ==0:
            get_extension = [s.split('.')[-1] for s in objn if '.'  in s ]
            if len(get_extension)==0 : get_extension=['']
            msg = ( "The available file types are {smart_format(get_extension)}"
                   if len(get_extension)!=0 else ''
                   ) 
            raise ValueError (f"None objects in the zip collection of matches"
                              f"the {ft!r}. Available file types are {msg}")
        return objn 
    
    if not os.path.isfile (zip_file ): 
        raise FileExistsError( f"File {os.path.basename(zip_file)!r} does"
                              " not exist. Expect a Path-like object,"
                              f" got {type(zip_file).__name__!r}")
        
    if not os.path.basename(zip_file ).lower().endswith ('.zip'): 
        raise FileNotFoundError("Unrecognized zip-file.")
        
    samples = str(samples) 
    if samples !='*': 
        try :samples = int (samples )
        except: 
            raise ValueError ("samples must be an integer value"
                              f" or '*' not {samples}")

    with ZipFile (zip_file, 'r', ) as zip_obj : 
        objnames = zip_obj.namelist() 
        if samples =='*':
                samples = len(objnames )
            
        if ftype is not None: 
            objnames = raise_msg_when(objn=objnames, ft= ftype) 

        if ( samples >= len(objnames) 
            and ftype is None
            ) :
            zip_obj.extractall( path = savepath , pwd=pwd) 
        else: 
            for zf in objnames [:samples ]: 
                zip_obj.extract ( zf, path = savepath, pwd = pwd)        
    
    return objnames 

def to_hdf5(d, fn, objname =None, close =True,  **hdf5_kws): 
    """
    Store a frame data in hierachical data format 5 (HDF5) 
    
    Note that is `d` is a dataframe, make sure that the dependency 'pytables'
    is already installed, otherwise and error raises. 
    
    Parameters 
    -----------
    d: ndarray, 
        data to store in HDF5 format 
    fn: str, 
        File path to HDF5 file.
    objname: str, 
        name of the data to store 
    close: bool, default =True 
        when data is given as an array, data can still be added if 
        close is set to ``False``, otherwise, users need to open again in 
        read mode 'r' before pursuing the process of adding. 
    hdf5_kws: dict of :class:`pandas.pd.HDFStore`  
        Additional keywords arguments passed to pd.HDFStore. they could be:
        *  mode : {'a', 'w', 'r', 'r+'}, default 'a'
    
             ``'r'``
                 Read-only; no data can be modified.
             ``'w'``
                 Write; a new file is created (an existing file with the same
                 name would be deleted).
             ``'a'``
                 Append; an existing file is opened for reading and writing,
                 and if the file does not exist it is created.
             ``'r+'``
                 It is similar to ``'a'``, but the file must already exist.
         * complevel : int, 0-9, default None
             Specifies a compression level for data.
             A value of 0 or None disables compression.
         * complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
             Specifies the compression library to be used.
             As of v0.20.2 these additional compressors for Blosc are supported
             (default if no compressor specified: 'blosc:blosclz'):
             {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
              'blosc:zlib', 'blosc:zstd'}.
             Specifying a compression library which is not available issues
             a ValueError.
         * fletcher32 : bool, default False
             If applying compression use the fletcher32 checksum.
    Returns
    ------- 
    store : Dict-like IO interface for storing pandas objects.
    
    Examples 
    ------------
    >>> import os 
    >>> from gofast.tools.coreutils import sanitize_frame_cols, to_hdf5 
    >>> from gofast.tools import read_data 
    >>> data = read_data('data/boreholes/H502.xlsx') 
    >>> sanitize_frame_cols (data, fill_pattern='_', inplace =True ) 
    >>> store_path = os.path.join('gofast/datasets/data', 'h') # 'h' is the name of the data 
    >>> store = to_hdf5 (data, fn =store_path , objname ='h502' ) 
    >>> store 
    ... 
    >>> # fetch the data 
    >>> h502 = store ['h502'] 
    >>> h502.columns[:3] 
    ... Index(['hole_number', 'depth_top', 'depth_bottom'], dtype='object')

    """
    from ._dependency import import_optional_dependency 
    
    store =None 
    if ( 
        not hasattr (d, '__array__') 
        or not hasattr (d, 'columns')
            ) : 
        raise TypeError ("Expect an array or dataframe,"
                         f" not {type (d).__name__!r}")
        
    if hasattr (d, '__array__') and hasattr (d, "columns"): 
        # assert whether pytables is installed 
        import_optional_dependency ('tables') 
        # remove extension if exist.
        fn = str(fn).replace ('.h5', "").replace(".hdf5", "")
        # then store. 
        store = pd.HDFStore(fn +'.h5' ,  **hdf5_kws)
        objname = objname or 'data'
        store[ str(objname) ] = d 

    
    elif not hasattr(d, '__array__'): 
        d = np.asarray(d) 
 
        store= h5py.File(f"{fn}.hdf5", "w") 
        store.create_dataset("dataset_01", store.shape, 
                             dtype=store.dtype,
                             data=store
                             )
        
    if close : store.close () 

    return store 
    
def store_or_write_hdf5 (
    d,  
    key:str= None, 
    mode:str='a',  
    kind: str=None, 
    path_or_buf:str= None, 
    encoding:str="utf8", 
    csv_sep: str=",",
    index: bool=..., 
    columns:Union [str, List[Any]]=None, 
    sanitize_columns:bool=...,  
    func: Optional[callable]= None, 
    args: tuple=(), 
    applyto: Union [str, List[Any]]=None, 
    **func_kwds, 
    )->Union [None, pd.DataFrame]: 
    """ Store data to hdf5 or write data to csv file. 
    
    Note that by default, the data is not store nor write and 
    return data if frame or transform the Path-Like object to data frame. 

    Parameters 
    -----------
    d: Dataframe, shape (m_samples, n_features)
        data to store or write or sanitize.
    key:str
       Identifier for the group in the store.
       
    mode: {'a', 'w', 'r+'}, default 'a'
       Mode to open file:
    
       - 'w': write, a new file is created (an existing file with the 
                                          same name would be deleted).
       - 'a': append, an existing file is opened for reading and writing, 
         and if the file does not exist it is created.
       - 'r+': similar to 'a', but the file must already exist.
       
    kind: str, {'store', 'write', None} , default=None 
       Type of task to perform: 
           
       - 'store': Store data to hdf5
       - 'write': export data to csv file.
       - None: construct a dataframe if array is passed or sanitize it. 

    path_or_buf: str or pandas.HDFStore, or str, path object, file-like \
        object, or None, default=None 
       File path or HDFStore object. String, path object
       (implementing os.PathLike[str]), or file-like object implementing 
       a write() function. If ``write=True`` and  None, the result is returned 
       as a string. If a non-binary file object is passed, it should be 
       opened with newline=" ", disabling universal newlines. If a binary 
       file object is passed, mode might need to contain a 'b'.
      
    encoding: str, default='utf8'
       A string representing the encoding to use in the output file, 
       Encoding is not supported if path_or_buf is a non-binary file object. 

    csv_sep: str, default=',', 
       String of length 1. Field delimiter for the output file.
       
    index: bool, index =False, 
       Write data to csv with index or not. 
       
    columns: list of str, optional 
        Usefull to create a dataframe when array is passed. Be aware to fit 
        the number of array columns (shape[1])
        
    sanitize_columns: bool, default=False, 
       remove undesirable character in the data columns using the default
       argument of `regex` parameters and fill pattern to underscore '_'. 
       The default regex implementation is:: 
           
           >>> import re 
           >>> re.compile (r'[_#&.)(*@!,;\s-]\s*', flags=re.IGNORECASE)
           
    func: callable, Optional 
       A custom sanitizing function and apply to each columns of the dataframe.
       If provide, the expected columns must be listed to `applyto` parameter.
       
    args: tuple, optional 
       Positional arguments of the sanitizing columns 
       
    applyto: str or list of str, Optional 
       The list of columns to apply the function ``func``. To apply the 
       function to all columns, use the ``*`` instead. 
       
    func_kwds: dict, 
       Keywords arguments of the sanitizing function ``func``. 
       
    Return 
    -------
    None or d: None of dataframe. 
      returns None if `kind` is set to ``write`` or ``store`` otherwise 
      return the dataframe. 
  
    Examples
    --------
    >>> from gofast.tools.ioutils import store_or_write_hdf5
    >>> from gofast.datasets import load_bagoue 
    >>> data = load_bagoue().frame 
    >>> data.geol[:5]
    0    VOLCANO-SEDIM. SCHISTS
    1                  GRANITES
    2                  GRANITES
    3                  GRANITES
    4          GEOSYN. GRANITES
    Name: geol, dtype: object
    >>> data = store_or_write_hdf5 ( data, sanitize_columns = True)
    >>> data[['type', 'geol', 'shape']] # put all to lowercase
      type                    geol shape
    0   cp  volcano-sedim. schists     w
    1   ec                granites     v
    2   ec                granites     v
    >>> # compute using func 
    >>> def test_func ( a, times  , to_percent=False ): 
            return ( a * times / 100)   if to_percent else ( a *times )
    >>> data.sfi[:5]
    0    0.388909
    1    1.340127
    2    0.446594
    3    0.763676
    4    0.068501
    Name: sfi, dtype: float64
    >>> d = store_or_write_hdf5 ( data,  func = test_func, args =(7,), applyto='sfi')
    >>> d.sfi[:5] 
    0    2.722360
    1    9.380889
    2    3.126156
    3    5.345733
    4    0.479507
    Name: sfi, dtype: float64
    >>> store_or_write_hdf5 ( data,  func = test_func, args =(7,),
                          applyto='sfi', to_percent=True).sfi[:5]
    0    0.027224
    1    0.093809
    2    0.031262
    3    0.053457
    4    0.004795
    Name: sfi, dtype: float64
    >>> # write data to hdf5 and outputs to current directory 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data.h5', 
                          kind ='store')
    >>> # export data to csv 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data', 
                          kind ='export')
    """
    # XXX revise imports 
    from .coreutils import to_numeric_dtypes,  exist_features  
    
    kind= key_search (str(kind), default_keys=(
        "none", "store", "write", "export", "tocsv"), 
        raise_exception=True , deep=True)[0]
    
    kind = "export" if kind in ('write', 'tocsv') else kind 
    
    if sanitize_columns is ...: 
        sanitize_columns=False 
    d = to_numeric_dtypes(d, columns=columns,sanitize_columns=sanitize_columns, 
                          fill_pattern='_')
   
    # get categorical variables 
    if ( sanitize_columns 
        or func is not None
        ): 
        d, _, cf = to_numeric_dtypes(d, return_feature_types= True )
        #( strip then pass to lower case all non-numerical data) 
        # for minimum sanitization  
        for cat in cf : 
            d[cat]= d[cat].str.lower()
            d[cat]= d[cat].str.strip()
            
    if func is not None: 
        if not callable(func): 
            raise TypeError(
                f"Expect a callable for `func`. Got {type(func).__name__!r}")

        if applyto is None:
            raise ValueError("Need to specify the data column to apply"
                             f"{func.__name__!r} to.")
        
        applyto = to_iterable( 
            applyto, exclude_string=True, transform =True 
            ) if applyto !="*" else d.columns 
        # check whether the applyto columns are in data columns 
        exist_features(d, applyto)
        
        # map each colum 
        for col in applyto: 
            d [col]=d[col].apply( func, args=args, **func_kwds )

    # store in h5 file. 
    if kind=='store':
        if path_or_buf is None: 
            print("Destination file is missing. Use 'data.h5' instead outputs"
                  f" in the current directory {os.getcwd()}")
            path_or_buf= 'data.h5'
 
        d.to_hdf ( path_or_buf , key =key, mode =mode )
    # export to csv file
    if kind=="export": 
        d.to_csv(path_or_buf, encoding = encoding  , sep=csv_sep , 
                 index =False if index is ... else index   )
        
    return d if kind not in ("store", "export") else None 

def key_checker (
    keys: str ,   
    valid_keys:List[str], 
    regex:re = None, 
    pattern:str = None , 
    deep_search:bool =...
    ): 
    """check whether a give key exists in valid_keys and return a list if 
    many keys are found.
    
    Parameters 
    -----------
    keys: str, list of str 
       Key value to find in the valid_keys 
       
    valid_keys: list 
       List of valid keys by default. 
       
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    deep_search: bool, default=False 
       If deep-search, the key finder is no sensistive to lower/upper case 
       or whether a numeric data is included. 
 
       
    Returns 
    --------
    keys: str, list , 
      List of keys that exists in the `valid_keys`. 
      
    Examples
    --------
    
    >>> from gofast.tools.coreutils import key_checker
    >>> key_checker('h502', valid_keys= ['h502', 'h253','h2601'])  
    Out[68]: 'h502'
    >>> key_checker('h502+h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[69]: ['h502', 'h2601']
    >>> key_checker('h502 h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[70]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2601'], valid_keys= ['h502', 'h253','h2601'])
    Out[73]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2602'], valid_keys= ['h502', 'h253','h2601'])
    UserWarning: key 'h2602' is missing in ['h502', 'h2602']
    Out[82]: 'h502'
    >>> key_checker(['502',  'H2601'], valid_keys= ['h502', 'h253','h2601'], 
                    deep_search=True )
    Out[57]: ['h502', 'h2601']
    
    """
    deep_search= False if deep_search ==... else True 

    _keys = copy.deepcopy(keys)
    valid_keys = to_iterable(valid_keys , exclude_string =True, transform =True )
    if isinstance ( keys, str): 
        pattern = pattern or '[_#&@!_+,;\s-]\s*'
        keys = str2columns (keys, regex = regex , pattern=pattern )
    # If iterbale object , save obj 
    # to improve error 
    kkeys = copy.deepcopy(keys)
    if deep_search: 
        keys = key_search(
            keys, 
            default_keys= valid_keys,
            deep=True, 
            raise_exception= True,
            regex =regex, 
            pattern=pattern 
            )
        return keys[0] if len(keys)==1 else keys 
    # for consistency 
    keys = [ k for k in keys if ''.join(
        [ str(i) for i in valid_keys] ).find(k)>=0 ]
    # assertion error if key does not exist. 
    if len(keys)==0: 
        verb1, verb2 = ('', 'es') if len(kkeys)==1 else ('s', '') 
        msg = (f"key{verb1} {_keys!r} do{verb2} not exist."
               f" Expect {' ,'.join(valid_keys)}") # dont use smartformat here 
        raise KeyError ( msg )
        
    if len(keys) != len(kkeys):
        # dont use is_in_if 
        miss_keys = is_in_if ( kkeys, keys , return_diff= True , error ='ignore')
        miss_keys, verb = (miss_keys[0], 'is') if len( miss_keys) ==1 else ( 
            miss_keys, 'are')
        warnings.warn(f"key{'' if verb=='is' else 's'} {miss_keys!r} {verb}"
                      f" missing in {_keys}")
    keys = keys[0] if len(keys)==1 else keys 
    
    return keys
 
def is_in_if (o: iter,  items: Union [str , iter], error = 'raise', 
               return_diff =False, return_intersect = False): 
    """ Raise error if item is not  found in the iterable object 'o' 
    
    :param o: unhashable type, iterable object,  
        object for checkin. It assumes to be an iterable from which 'items' 
        is premused to be in. 
    :param items: str or list, 
        Items to assert whether it is in `o` or not. 
    :param error: str, default='raise'
        raise or ignore error when none item is found in `o`. 
    :param return_diff: bool, 
        returns the difference items which is/are not included in 'items' 
        if `return_diff` is ``True``, will put error to ``ignore`` 
        systematically.
    :param return_intersect:bool,default=False
        returns items as the intersection between `o` and `items`.
    :raise: ValueError 
        raise ValueError if `items` not in `o`. 
    :return: list,  
        `s` : object found in ``o` or the difference object i.e the object 
        that is not in `items` provided that `error` is set to ``ignore``.
        Note that if None object is found  and `error` is ``ignore`` , it 
        will return ``None``, otherwise, a `ValueError` raises. 
        
    :example: 
        >>> from gofast.datasets import load_hlogs 
        >>> from gofast.tools.coreutils import is_in_if 
        >>> X0, _= load_hlogs (as_frame =True )
        >>> is_in_if  (X0 , items= ['depth_top', 'top']) 
        ... ValueError: Item 'top' is missing in the object 
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore') 
        ... ['depth_top']
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore',
                       return_diff= True) 
        ... ['sp',
         'well_diameter',
         'layer_thickness',
         'natural_gamma',
         'short_distance_gamma',
         'strata_name',
         'gamma_gamma',
         'depth_bottom',
         'rock_name',
         'resistivity',
         'hole_id']
    """
    
    if isinstance (items, str): 
        items =[items]
    elif not to_iterable(o): 
        raise TypeError (f"Expect an iterable object, not {type(o).__name__!r}")
    # find intersect object 
    s= set (o).intersection (items) 
    
    miss_items = list(s.difference (o)) if len(s) > len(
        items) else list(set(items).difference (s)) 

    if return_diff or return_intersect: 
        error ='ignore'
    
    if len(miss_items)!=0 :
        if error =='raise': 
            v= ','.join(miss_items) # use ','. join instead of smart_format
            verb = f"{ ' '+ v +' is' if len(miss_items)<2 else  's '+ v + 'are'}"
            raise ValueError (
                f"Item{verb} missing in the {type(o).__name__.lower()} {o}.")
            
       
    if return_diff : 
        # get difference 
        s = list(set(o).difference (s))  if len(o) > len( 
            s) else list(set(items).difference (s)) 
        # s = set(o).difference (s)  
    elif return_intersect: 
        s = list(set(o).intersection(s))  if len(o) > len( 
            items) else list(set(items).intersection (s))     
    
    s = None if len(s)==0 else list (s) 
    
    return s  

def str2columns (text,  regex=None , pattern = None): 
    """Split text from the non-alphanumeric markers using regular expression. 
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    text: str, 
        text litteral containing the columns the names to retrieve
        
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[#&*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import str2columns 
    >>> text = ('this.is the text to split. It is an: example of; splitting str - to text.')
    >>> str2columns (text )  
    ... ['this',
         'is',
         'the',
         'text',
         'to',
         'split',
         'It',
         'is',
         'an:',
         'example',
         'of',
         'splitting',
         'str',
         'to',
         'text']

    """
    pattern = pattern or  r'[#&.*@!_,;\s-]\s*'
    regex = regex or re.compile (pattern, flags=re.IGNORECASE) 
    text= list(filter (None, regex.split(str(text))))
    return text 

def key_search (
    keys: str,  
    default_keys: Union [Text , List[str]], 
    parse_keys: bool=True, 
    regex :re=None, 
    pattern :str=None, 
    deep: bool =...,
    raise_exception:bool=..., 
    ): 
    """Find key in a list of default keys and select the best match. 
    
    Parameters 
    -----------
    keys: str or list 
       The string or a list of key. When multiple keys is passed as a string, 
       use the space for key separating. 
       
    default_keys: str or list 
       The likehood key to find. Can be a litteral text. When a litteral text 
       is passed, it is better to provide the regex in order to skip some 
       character to parse the text properly. 
       
    parse_keys: bool, default=True 
       Parse litteral string using default `pattern` and `regex`. 
       
       .. versionadded:: 0.2.7 
        
    regex: `re` object,  
        Regular expresion object. Regex is important to specify the kind
        of data to parse. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns. Pattern is 
        important especially when some character are considers as a part of 
        word but they are not a separator. For example a data columns with 
        a name `'DH_Azimuth'`, if a pattern is not explicitely provided, 
        the default pattern will parse as two separated word which is far 
        from the expected results. 
        
    deep: bool, default=False 
       Not sensistive to uppercase. 
       
    raise_exception: bool, default=False 
       raise error when key is not find. 
       
    Return 
    -------
    list: list of valid keys or None if not find ( default) 

    Examples
    ---------
    >>> from gofast.tools.coreutils import key_search 
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'])
    Out[44]: ['h502']
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'], 
                   deep=True)
    Out[46]: ['h502', 'HH2601']
    >>> key_search('253', default_keys= ("I m here to find key among h502,
                                             h253 and HH2601"))
    Out[53]: ['h253'] 
    >>> key_search ('east', default_keys= ['DH_East', 'DH_North']  , deep =True,)
    Out[37]: ['East']
    key_search ('east', default_keys= ['DH_East', 'DH_North'], 
                deep =True,parse_keys= False)
    Out[39]: ['DH_East']
    """
    def _ellipsis2false ( param): 
        if param ==...:
            return False 
        return True 
    deep, raise_exception, parse_keys = [_ellipsis2false(
        param) for param in [deep, raise_exception, parse_keys] ] 
    # make a copy of original keys 
    
    kinit = copy.deepcopy(keys)
    if parse_keys: 
        if to_iterable(keys , exclude_string= True ): 
            keys = ' '.join ( [str(k) for k in keys ]) 
             # for consisteny checker 
        pattern = pattern or '[#&@!_+,;\s-]\s*'
        keys = str2columns ( keys , regex = regex , pattern = pattern ) 
            
        if to_iterable ( default_keys , exclude_string=True ): 
            default_keys = ' '. join ( [ str(k) for k in default_keys ])
            # make a copy
        default_keys =  str2columns(
            default_keys, regex =regex , pattern = pattern )
    else : 
        keys = to_iterable(
        keys, exclude_string = True, transform =True )
        default_keys = to_iterable ( 
            default_keys, exclude_string=True, transform =True )
        
    dk_init = copy.deepcopy(default_keys )
    # if deep convert all keys to lower 
    if deep: 
        keys= [str(it).lower() for it in keys  ]
        default_keys = [str(it).lower() for it in default_keys  ]

    valid_keys =[] 
    for key in keys : 
        for ii, dkey in enumerate (default_keys) : 
            vk = re.findall(rf'\w*{key}\w*', dkey)
            # rather than rf'\b\w*{key}\w*\b'
            # if deep take the real values in defaults keys.
            if len(vk) !=0: 
                if deep: valid_keys.append( dk_init[ii] )
                else:valid_keys.extend( vk)
                break     
    if ( raise_exception 
        and len(valid_keys)==0
        ): 
        kverb ='s' if len(kinit)> 1 else ''
        raise KeyError (f"key{kverb} {kinit!r} not found."
                       f" Expect {','.join(dk_init)}")
    return None if len(valid_keys)==0 else valid_keys 


def serialize_data(
    data,
    filename: str = None,
    force: bool = True,
    savepath: str = None,
    verbose: int = 0
) -> str:
    """
    Serializes data to a binary file using joblib or pickle.

    Parameters
    ----------
    data : Any
        The data object to serialize.
    filename : str, optional
        Filename for the serialized data. If None, a filename with timestamp 
        is generated.
    force : bool, default=True
        If True, overwrites existing files. If False, creates a unique file 
        by appending the timestamp.
    savepath : str, optional
        Directory in which to save the file. If not specified, saves to 
        the current working directory.
    verbose : int, default=0
        Controls verbosity of output messages.

    Returns
    -------
    str
        The path to the serialized file.

    Examples
    --------
    >>> data = [1, 2, 3]
    >>> serialize_data(data, filename='data.pkl', force=True)
    'path/to/data.pkl'
    """
    if filename is None:
        filename = f"serialized_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    
    filepath = os.path.join(savepath or os.getcwd(), filename)
    
    if os.path.exists(filepath) and not force:
        filename = filename.replace(
            '.pkl', f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
        filepath = os.path.join(savepath or os.getcwd(), filename)
        
    try:
        joblib.dump(data, filepath)
        if verbose > 0:
            print(f"Data serialized to {filepath}")
    except Exception as e: # noqa 
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        if verbose > 0:
            print(f"Data serialized using pickle to {filepath}")
    
    return filepath


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
    >>> from gofast.tools.coreutils import replace_data
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

def save_path(nameOfPath: str) -> str:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    nameOfPath : str
        Name or path of the directory to create.

    Returns
    -------
    str
        The path of the created directory. If it exists, returns the existing path.

    Examples
    --------
    >>> save_path("test_directory")
    'path/to/test_directory'
    """
    path = os.path.join(os.getcwd(), nameOfPath)
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_unicode_string(str_: str) -> str:
    """
    Removes spaces and replaces accented characters in a string.

    Parameters
    ----------
    str_ : str
        The string to sanitize.

    Returns
    -------
    str
        The sanitized string with removed spaces and replaced accents.

    Examples
    --------
    >>> from gofast.tools.ioutils import sanitize_unicode_string 
    >>> sentence ='Nos clients sont extrêmement satisfaits '
        'de la qualité du service fourni. En outre Nos clients '
            'rachètent frequemment nos "services".'
    >>> sanitize_unicode_string  (sentence)
    ... 'nosclientssontextrmementsatisfaitsdelaqualitduservice'
        'fournienoutrenosclientsrachtentfrequemmentnosservices'
    >>> sanitize_unicode_string("Élève à l'école")
    'elevealecole'
    """
    accents_replacements = {'éèê': 'e', 'àâ': 'a'}
    str_ = re.sub(r'\s+', '', str_.lower())
    
    for chars, repl in accents_replacements.items():
        str_ = re.sub(f"[{chars}]", repl, str_)
    
    return str_

def parse_md_data(pf: str, delimiter: str = ':'):
    """
    Parse a markdown-style file with key-value pairs separated by 
    a delimiter.

    Parameters
    ----------
    pf : str
        Path to the markdown file containing key-value pairs.
    delimiter : str, default=':'
        Delimiter used to separate key-value pairs.

    Yields
    ------
    Tuple[str, str]
        A tuple containing the key and processed value.

    Raises
    ------
    IOError
        If the provided path does not lead to a valid file.

    Notes
    -----
    - This function yields key-value pairs by reading the file line-by-line.
    - It applies `sanitize_unicode_string` to keys to ensure data consistency.

    Examples
    --------
    >>> list(parse_md_data('parser_file.md', delimiter=':'))
    [('key1', 'Value1'), ('key2', 'Value2')]
    """
    if not os.path.isfile(pf):
        raise IOError("Unable to detect the parser file. Need a Path-like object.")

    with open(pf, 'r', encoding='utf8') as f:
        pdata = f.readlines()

    for row in pdata:
        if row in ('\n', ' '):
            continue
        fr, en = row.strip().split(delimiter)
        fr = sanitize_unicode_string(fr)  # Clean up the key
        en = en.strip()
        
        # Capitalize the first letter of the value
        en = en[0].upper() + en[1:]
        
        yield fr, en


def dummy_csv_translator(
        csv_fn: str, pf: str, delimiter: str = ':',
        destfile: str = 'pme.en.csv'):
    """
    Translate a CSV file using a dictionary created from a markdown-style parser file.

    Parameters
    ----------
    csv_fn : str
        Path to the source CSV file.
    pf : str
        Path to the markdown-style file used to create the translation dictionary.
    delimiter : str, default=':'
        Delimiter used in the parser file to separate key-value pairs.
    destfile : str, default='pme.en.csv'
        Name of the destination file for the translated CSV.

    Returns
    -------
    DataFrame
        Translated CSV data as a DataFrame.
    list
        List of untranslated terms found in the source CSV.

    Notes
    -----
    - This function uses `parse_md_data` to read the parser file and apply
      translations to the CSV content.
    - Missing translations are collected and returned for review.

    Examples
    --------
    >>> df, missing = dummy_csv_translator(
        "data.csv", "parser_file.md", delimiter=":", destfile="output.csv")
    >>> print(df.head())
    >>> print(missing)

    """
    parser_data = dict(parse_md_data(pf, delimiter))
    
    # Read CSV data
    with open(csv_fn, 'r', encoding='utf8') as csv_f:
        csv_reader = csv.reader(csv_f)
        csv_data = [row for row in csv_reader]

    # Locate 'Industry_type' column and split data blocks
    industry_index = csv_data[0].index('Industry_type')
    csv_1b = [row[:industry_index + 1] for row in csv_data]
    csv_2b = [row[industry_index + 1:] for row in csv_data]

    # Clean data in `csv_1b` and collect missing translations
    csv_1b_cleaned = copy.deepcopy(csv_1b)
    untranslated_terms = set()
    for row in csv_1b_cleaned[3:]:
        for i, value in enumerate(row):
            value = sanitize_unicode_string(value.strip())
            if value not in parser_data:
                untranslated_terms.add(value)
            else:
                row[i] = parser_data.get(value, value)

    # Combine cleaned blocks and convert to DataFrame
    combined_data = [r1 + r2 for r1, r2 in zip(csv_1b_cleaned, csv_2b)]
    df = pd.DataFrame(np.array(combined_data[1:]), columns=combined_data[0])

    # Apply parser dictionary and save to destination file
    df.replace(parser_data, inplace=True)
    df.to_csv(destfile, index=False)
    
    return df, list(untranslated_terms)


def rename_files(
    src_files: Union[str, List[str]], 
    dst_files: Union[str, List[str]], 
    basename: Optional[str] = None, 
    extension: Optional[str] = None, 
    how: str = 'py', 
    prefix: bool = True, 
    keep_copy: bool = True, 
    trailer: str = '_', 
    sortby: Union[re.Pattern, callable] = None, 
    **kws
) -> None:
    """Rename files in directory.

    Parameters 
    -----------
    src_files: str, Path-like object 
       Source files to rename 
      
    dst_files: str of PathLike object 
       Destination files renamed. 
       
    extension: str, optional 
       If a path is given in `src_files`, specifying the `extension` will just 
       collect only files with this typical extensions. 
       
    basename: str, optional 
       If `dst_files` is passed as Path-object, name should be needed 
       for a change, otherwise, the number is incremented using the Python 
       index counting defined by the parameter ``how=py` 
        
    how: str, default='py' 
       The way to increment files when `dst_files` is given as a Path object. 
       For instance, for a  ``name=E_survey`` and ``prefix==True``, the first 
       file should be ``E_survey_00`` if ``how='py'`` otherwise it should be 
       ``E_survey_01``.
     
    prefix: bool, default=True
      Prefix is used to position the name before the number incrementation. 
      If ``False`` and `name` is given, the number is positionning before the 
      name. If ``True`` and not `prefix` for a ``name=E_survey``, it should be 
      ``00_E_survey`` and ``01_E_survey``. 

    keep_copy: bool, default=True 
       Keep a copy of the source files. 
       
    trailer: str, default='_', 
       Item used to separate the basename for counter. 
       
    sortby: Regex or Callable, 
       Key to sort the collection of the items when `src_files` is passed as 
       a path-like object.  This is usefull to keep order as the origin files 
       especially  when files includes a specific character.  Furthermore 
       [int| float |'num'|'digit'] sorted the files according to the
       number included in the filename if exists. 

    kws: dict 
       keyword arguments passed to `os.rename`. 

    """ 
    dest_dir =None ; trailer = str(trailer)
    extension = str(extension).lower()
    
    if os.path.isfile (src_files ): 
        src_files = [src_files ] 
        
    elif os.path.isdir (src_files): 
        src_path = src_files
        ldir = os.listdir(src_path) 

        src_files = ldir if extension =='none' else [
             f for f in ldir  if  f.endswith (extension) ]
    
        if sortby: 
            if sortby in ( int, float, 'num', 'number', 'digit'): 
                src_files = sorted(ldir, key=lambda s:int( re.search(
                    '\d+', s).group()) if re.search('\d+', s) else 0 )
            else: 
                src_files = sorted(ldir, key=sortby)

        src_files = [  os.path.join(src_path, f )   for f in src_files  ] 
        # get only the files 
        src_files = [ f for f in src_files if os.path.isfile (f ) ]

    else : raise FileNotFoundError(f"{src_files!r} not found.") 
    
    # Create the directory if it doesn't exist
    if ( dst_files is not None 
        and not os.path.exists (dst_files)
        ): 
        os.makedirs(dst_files)
        
    if os.path.isdir(dst_files): 
        dest_dir = dst_files 
    
    if isinstance (dst_files, str): 
        dst_files = [dst_files]
        #XXX revise 
   # dst_files = is_iterable(dst_files , exclude_string= True, transform =True ) 
    # get_extension of the source_files 
    _, ex = os.path.splitext (src_files[0]) 
    
    if dest_dir: 
        if basename is None: 
            warnings.warn(
                "Missing basename for renaming file. Should use `None` instead.")
            basename =''; trailer =''
            
        basename= str(basename)
        if prefix: 
            dst_files =[ f"{str(basename)}{trailer}" + (
                f"{i:03}" if how=='py' else f"{i+1:03}") + f"{ex}"
                        for i in range (len(src_files))]
        elif not prefix: 
            dst_files =[ (f"{i:03}" if how=='py' else f"{i+1:03}"
                        ) +f"{trailer}{str(basename)}" +f"{ex}"
                        for i in range (len(src_files))]
        
        dst_files = [os.path.join(dest_dir , f) for f in dst_files ] 
    
    for f, nf in zip (src_files , dst_files): 
        try: 
           if keep_copy : shutil.copy (f, nf , **kws )
           else : os.rename (f, nf , **kws )
        except FileExistsError: 
            os.remove(nf)
            if keep_copy : shutil.copy (f, nf , **kws )
            else : os.rename (f, nf , **kws )
            
def cpath(savepath: str = None, dpath: str = '_default_path_') -> str:
    """
    Ensures a directory exists for saving files, creating it if necessary.

    Parameters
    ----------
    savepath : str, optional
        The target directory to validate or create. If None, `dpath` is used 
        as the directory.

    dpath : str, default='_default_path_'
        Default directory created in the current working directory if 
        `savepath` is None.

    Returns
    -------
    str
        The absolute path to the validated or created directory.

    Examples
    --------
    >>> from gofast.tools.ioutils import cpath
    >>> default_path = cpath()
    >>> print(f"Files will be saved to: {default_path}")

    >>> custom_path = cpath('/path/to/save')
    >>> print(f"Files will be saved to: {custom_path}")

    Notes
    -----
    `cpath` validates the directory path and, if necessary, creates the
    directory tree. If a problem occurs during creation, an error message 
    is printed.

    See Also
    --------
    pathlib.Path.mkdir : Utility for directory creation.
    """
    if savepath is None:
        # Use default directory path if none provided
        savepath = Path.cwd() / dpath
    else:
        savepath = Path(savepath)

    try:
        # Create the directory and parents if they do not exist
        savepath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {savepath}: {e}")

    return str(savepath.resolve())


def spath(name_of_path: str) -> str:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    name_of_path : str
        Path-like object to create if it doesn't exist.

    Returns
    -------
    str
        The absolute path to the created or existing directory.

    Examples
    --------
    >>> from gofast.tools.ioutils import spath
    >>> path = spath('data/saved_models')
    >>> print(f"Directory available at: {path}")

    Notes
    -----
    `spath` is useful for quickly ensuring that a specific directory is 
    available for storing files. It provides feedback if the directory
    already exists.
    """
    savepath = os.path.join(os.getcwd(), name_of_path)
    try:
        if not os.path.isdir(savepath):
            os.mkdir(name_of_path)
    except:
        warnings.warn("The path already exists.")
    return savepath


def load_serialized_data(filename: str, verbose: int = 0):
    """
    Load data from a serialized file (e.g., pickle or joblib format).

    Parameters
    ----------
    filename : str
        Name of the file to load data from.

    verbose : int, default=0
        Verbosity level. Controls the amount of output information:
        - 0: No output
        - >2: Detailed loading process messages.

    Returns
    -------
    Any
        Data loaded from the file, or None if deserialization fails.

    Raises
    ------
    TypeError
        If `filename` is not a string.

    FileExistsError
        If the specified file does not exist.

    Examples
    --------
    >>> from gofast.tools.ioutils import load_serialized_data
    >>> data = load_serialized_data('data/my_data.pkl', verbose=3)

    Notes
    -----
    This function attempts to load serialized data using joblib and 
    fallbacks to pickle if needed. Verbose output provides feedback on 
    the loading process and success or failure of each step.

    See Also
    --------
    joblib.load : High-performance loading utility.
    pickle.load : General-purpose Python serialization library.
    """
    if not isinstance(filename, str):
        raise TypeError(f"filename should be a <str> not <{type(filename)}>")

    if not os.path.isfile(filename):
        raise FileExistsError(f"File {filename!r} does not exist.")

    _filename = os.path.basename(filename)
    data = None

    try:
        # Attempt to load with joblib
        data = joblib.load(filename)
        if verbose > 2:
            print(f"Data from {_filename!r} successfully reloaded using joblib.")
    except:
        if verbose > 2:
            print(f"Fallback: {_filename!r} not loaded with joblib; trying pickle.")
        with open(filename, 'rb') as tod:
            data = pickle.load(tod)
        if verbose > 2:
            print(f"Data from {_filename!r} reloaded using pickle.")

    if verbose > 0:
        if data is None:
            print("Unable to deserialize data. Please check your file.")
        else:
            print(f"Data from {_filename} has been successfully reloaded.")

    return data


def save_job(
    job , 
    savefile ,* ,  
    protocol =None,  
    append_versions=True, 
    append_date=True, 
    fix_imports= True, 
    buffer_callback = None,   
    **job_kws
    ): 
    """ Quick save your job using 'joblib' or persistent Python pickle module.
    
    Parameters 
    -----------
    job: Any 
        Anything to save, preferabaly a models in dict 
        
    savefile: str, or path-like object 
         name of file to store the model
         The *file* argument must have a write() method that accepts a
         single bytes argument. It can thus be a file object opened for
         binary writing, an io.BytesIO instance, or any other custom
         object that meets this interface.
         
    append_versions: bool, default =True 
        Append the version of Joblib module or Python Pickle module following 
        by the scikit-learn, numpy and also pandas versions. This is useful 
        to have idea about previous versions for loading file when system or 
        modules have been upgraded. This could avoid bottleneck when data 
        have been stored for long times and user has forgotten the date and 
        versions at the time the file was saved. 
        
    append_date: bool, default=True, 
       Append the date  of the day to the filename. 
       
    protocol: int, optional 
        The optional *protocol* argument tells the pickler to use the
        given protocol; supported protocols are 0, 1, 2, 3, 4 and 5.
        The default protocol is 4. It was introduced in Python 3.4, and
        is incompatible with previous versions.
    
        Specifying a negative protocol version selects the highest
        protocol version supported.  The higher the protocol used, the
        more recent the version of Python needed to read the pickle
        produced.
        
    fix_imports: bool, default=True, 
        If *fix_imports* is True and *protocol* is less than 3, pickle
        will try to map the new Python 3 names to the old module names
        used in Python 2, so that the pickle data stream is readable
        with Python 2.
        
    buffer_call_back: int, optional 
        If *buffer_callback* is None (the default), buffer views are
        serialized into *file* as part of the pickle stream.
    
        If *buffer_callback* is not None, then it can be called any number
        of times with a buffer view.  If the callback returns a false value
        (such as None), the given buffer is out-of-band; otherwise the
        buffer is serialized in-band, i.e. inside the pickle stream.
    
        It is an error if *buffer_callback* is not None and *protocol*
        is None or smaller than 5.
        
    job_kws: dict, 
        Additional keywords arguments passed to :func:`joblib.dump`. 
        
    Returns
    -------
    str
        The final filename where the job was saved.

    Notes
    -----
    This function appends system-specific metadata like versions and date to
    the filename, which can aid in tracking compatibility over time.

    Examples
    --------
    >>> from gofast.tools.ioutils import save_job
    >>> model = {"key": "value"}  # Replace with actual model object
    >>> savefile = save_job(model, "my_model", append_date=True, append_versions=True)
    >>> print(savefile)
    'my_model.20240101.sklearn_v1.0.numpy_v1.21.joblib'

    """
    def remove_extension(filename: str, extension: str) -> str:
        return filename.replace(extension, '')

    import sklearn

    # Generate versioning metadata
    versions = 'sklearn_v{0}.numpy_v{1}.pandas_v{2}'.format(
        sklearn.__version__, np.__version__, pd.__version__)
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    # Handle file extensions
    savefile = str(savefile)
    extension = '.joblib' if '.joblib' in savefile else '.pkl'
    savefile = remove_extension(savefile, extension)

    # Append date and versions if requested
    if append_date:
        savefile += f".{date_str}"
    if append_versions:
        savefile += f".{versions}"

    try:
        joblib.dump(job, f"{savefile}.joblib", **job_kws)
    except Exception:
        with open(f"{savefile}.pkl", 'wb') as wfile:
            pickle.dump(job, wfile, protocol=protocol, 
                        fix_imports=fix_imports, 
                        buffer_callback=buffer_callback)

    return savefile

def cparser_manager(
    cfile: str,
    savepath: Optional[str] = None, 
    todo: str = 'load', 
    dpath: Optional[str] = None,
    verbose: int = 0, 
    **pkws
) -> None:
    """
    Manages configuration file saving and output messages based on action type.

    Parameters
    ----------
    cfile : str
        Name of the configuration file.
    savepath : str, optional
        Directory path to save the configuration file.
    todo : str, default='load'
        Action to perform with the config file. Options are 'load' or 'dump'.
    dpath : str, optional
        Default path to use if savepath is not specified.
    verbose : int, default=0
        Controls verbosity level of output messages.

    Notes
    -----
    This function uses `move_cfile` to ensure the configuration file is stored 
    in the correct location, and calls `print_cmsg` to provide user feedback.

    """
    if savepath == 'default':
        savepath = None
    yml_fn, _ = move_cfile(cfile, savepath, dpath=dpath)

    if verbose > 0:
        print(print_cmsg(yml_fn, todo, **pkws))

def move_cfile(
    cfile: str, 
    savepath: Optional[str] = None, 
    **ckws
) -> Tuple[str, str]:
    """
    Moves a file to the specified path. If moving fails, copies and 
    deletes the original.

    Parameters
    ----------
    cfile : str
        Name of the file to move.
    savepath : str, optional
        Target directory. If not specified, uses default path via `cpath`.

    Returns
    -------
    Tuple[str, str]
        The new file path and a confirmation message.

    Examples
    --------
    >>> from gofast.tools.ioutils import move_cfile
    >>> new_path, msg = move_cfile('myfile.txt', 'new_directory')
    >>> print(new_path, msg)

    """
    savepath = cpath(savepath or '_default_path_', **ckws)
    destination_file_path = os.path.join(savepath, os.path.basename(cfile))

    try:
        shutil.move(cfile, destination_file_path)
    except shutil.Error:
        shutil.copy2(cfile, destination_file_path)
        os.remove(cfile)

    msg = (f"--> '{os.path.basename(destination_file_path)}'successfully"
           f" saved to '{os.path.realpath(destination_file_path)}'."
           )
    return destination_file_path, msg

def print_cmsg(
        cfile: str, todo: str = 'load', config: str = 'YAML') -> str:
    """
    Generates output message for configuration file operations.

    Parameters
    ----------
    cfile : str
        Name of the configuration file.
    todo : str, default='load'
        Operation performed ('load' or 'dump').
    config : str, default='YAML'
        Type of configuration file (e.g., 'YAML', 'CSV', 'JSON').

    Returns
    -------
    str
        Confirmation message for the configuration operation.

    Examples
    --------
    >>> from gofast.tools.ioutils import print_cmsg
    >>> msg = print_cmsg('config.yml', 'dump')
    >>> print(msg)
    --> YAML 'config.yml' data was successfully saved.

    """
    if todo == 'load':
        msg = f"--> Data successfully loaded from '{os.path.realpath(cfile)}'."
    elif todo == 'dump':
        msg =( 
            f"--> {config.upper()} '{os.path.basename(cfile)}'"
            " data was successfully saved."
            )
    return msg



def parse_csv(
    csv_fn: str = None,
    data: Optional[Union[List[Dict], List[List[str]]]] = None,
    todo: str = 'reader', 
    fieldnames: Optional[List[str]] = None,
    savepath: Optional[str] = None,
    header: bool = False,
    verbose: int = 0,
    **csvkws
) -> Union[List[Dict], List[List[str]], None]:
    """
    Parses a CSV file or serializes data to a CSV file.

    This function allows loading (reading) from or dumping (writing) to a CSV 
    file. It supports standard CSV and dictionary-based CSV formats.

    Parameters
    ----------
    csv_fn : str, optional
        The CSV filename for reading or writing. For writing operations, if 
        `data` is provided and `todo` is set to 'write' or 'dictwriter', this 
        specifies the output CSV filename.
    data : list, optional
        Data to write in the form of a list of lists or dictionaries.
    todo : str, default='reader'
        Specifies the operation type:
        - 'reader' or 'dictreader': Reads data from a CSV file.
        - 'writer' or 'dictwriter': Writes data to a CSV file.
    fieldnames : list of str, optional
        List of keys for dictionary-based writing to specify the field order.
    savepath : str, optional
        Directory to save the CSV file when writing. Defaults to '_savecsv_' 
        if not provided and the path does not exist.
    header : bool, default=False
        If True, includes headers when writing with DictWriter.
    verbose : int, default=0
        Controls the verbosity level for output messages.
    csvkws : dict, optional
        Additional arguments passed to `csv.writer` or `csv.DictWriter`.

    Returns
    -------
    Union[List[Dict], List[List[str]], None]
        Parsed data from the CSV file, as a list of lists or a list of 
        dictionaries, based on the operation. Returns `None` when writing.

    Notes
    -----
    For writing data, the method uses either `csv.writer` for regular CSV or 
    `csv.DictWriter` for dictionary-based CSV depending on the value of `todo`.

    Examples
    --------
    >>> from gofast.tools.ioutils import parse_csv
    >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    >>> parse_csv(csv_fn='output.csv', data=data, todo='dictwriter', fieldnames=['name', 'age'])
    >>> loaded_data = parse_csv(csv_fn='output.csv', todo='dictreader', fieldnames=['name', 'age'])
    >>> print(loaded_data)
    [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]

    """
    todo, domsg = return_ctask(todo)

    if 'write' in todo:
        csv_fn = get_config_fname_from_varname(data, config_fname=csv_fn, config='.csv')

    try:
        if todo == 'reader':
            with open(csv_fn, 'r') as csv_f:
                csv_reader = csv.reader(csv_f)
                data = [row for row in csv_reader]
        elif todo == 'writer':
            with open(f"{csv_fn}.csv", 'w', newline='', encoding='utf8') as new_csvf:
                csv_writer = csv.writer(new_csvf, **csvkws)
                csv_writer.writerows(data) if len(data) > 1 else csv_writer.writerow(data)
        elif todo == 'dictreader':
            with open(csv_fn, 'r', encoding='utf8') as csv_f:
                csv_reader = csv.DictReader(csv_f, fieldnames=fieldnames)
                data = list(csv_reader)
        elif todo == 'dictwriter':
            with open(f"{csv_fn}.csv", 'w') as new_csvf:
                csv_writer = csv.DictWriter(new_csvf, fieldnames=fieldnames, **csvkws)
                if header:
                    csv_writer.writeheader()
                if isinstance(data, dict):
                    csv_writer.writerow(data)
                else:
                    csv_writer.writerows(data)
    except csv.Error as e:
        raise csv.Error(f"Unable {domsg} CSV {csv_fn!r}. {e}")
    except Exception as e:
        msg = "Unrecognizable file" if 'read' in todo else "Unable to write"
        raise TypeError(f"{msg} {csv_fn!r}. Check your"
                        f" {'file' if 'read' in todo else 'data'}. {e}")

    cparser_manager(f"{csv_fn}.csv", savepath, todo=todo, dpath='_savecsv_',
                    verbose=verbose, config='CSV')
    return data

def return_ctask(todo: Optional[str] = None) -> Tuple[str, str]:
    """
    Determine the action to perform based on the `todo` input.

    Parameters
    ----------
    todo : str, optional
        Specifies the action. Options:
        - 'load': Load data from a config file (YAML|CSV|JSON).
        - 'dump': Serialize data to a config file (YAML|CSV|JSON).

    Returns
    -------
    Tuple[str, str]
        `todo`: Corrected action string.
        `domsg`: Message for user based on action.

    Notes
    -----
    This function normalizes user input for `todo` to avoid misinterpretations.

    """
    def p_csv(v, cond='dict', base='reader'):
        return f"{cond}{base}" if cond in v else base

    ltags = ('load', 'recover', True, 'fetch')
    dtags = ('serialized', 'dump', 'save', 'write', 'serialize')
    if todo is None:
        raise ValueError(
            "NoneType action cannot be performed. Specify 'load' or 'dump'.")

    todo = str(todo).lower()
    ltags += ('loads',) if todo == 'loads' else ()
    dtags += ('dumps',) if todo == 'dumps' else ()

    if todo in ltags:
        todo, domsg = 'load', 'to parse'
    elif todo in dtags:
        todo, domsg = 'dump', 'to serialize'
    elif 'read' in todo:
        todo, domsg = p_csv(todo), 'to read'
    elif 'write' in todo:
        todo, domsg = p_csv(todo, base='writer'), 'to write'
    else:
        raise ValueError(
            f"Invalid action '{todo}'. Use 'load' or 'dump' (YAML|CSV|JSON).")

    return todo, domsg


def parse_yaml(
    yml_fn: str = None,
    data=None,
    todo: str = 'load',
    savepath: Optional[str] = None,
    verbose: int = 0,
    **ymlkws
):
    """
    Parse and handle YAML configuration files for loading or saving data.

    Parameters
    ----------
    yml_fn : str, optional
        The YAML filename. If `data` is provided and `todo` is set to 'dump',
        `yml_fn` will be used as the output filename. If `todo` is set to 
        'load', `yml_fn` is the input filename to read from.

    data : Any, optional
        Data in a Python object format that will be serialized and saved as a 
        YAML file if `todo` is 'dump'.

    todo : {'load', 'dump'}, default='load'
        Action to perform with the YAML file:
        - 'load': Load data from the YAML file specified by `yml_fn`.
        - 'dump': Serialize `data` into a YAML format and save to `yml_fn`.

    savepath : str, optional
        Path where the YAML file will be saved if `todo` is 'dump'. If not 
        provided, a default path will be used. The function will ensure that 
        the path exists.

    verbose : int, default=0
        Controls verbosity of output messages.

    **ymlkws : dict
        Additional keyword arguments passed to `yaml.dump` when saving data.

    Returns
    -------
    Any
        The data loaded from the YAML file if `todo` is 'load', or `data` 
        after saving if `todo` is 'dump'.

    Raises
    ------
    yaml.YAMLError
        If there is an issue with reading or writing the YAML file.

    Notes
    -----
    This function uses `safe_load` and `safe_dump` methods from PyYAML for 
    secure handling of YAML files.

    See Also
    --------
    `get_config_fname_from_varname` : Utility for generating YAML configuration 
    filenames based on variable names.
    """
    # Determine task for loading or dumping YAML
    todo = todo.lower()
    if todo.startswith('dump'):
        yml_fn = get_config_fname_from_varname(data, yml_fn)
        try:
            with open(f"{yml_fn}.yml", "w") as fw:
                yaml.safe_dump(data, fw, **ymlkws)
        except yaml.YAMLError:
            raise yaml.YAMLError(
                f"Unable to save data to {yml_fn}. Check file permissions.")
    elif todo.startswith('load'):
        try:
            with open(yml_fn, "r") as fy:
                data = yaml.safe_load(fy)
        except yaml.YAMLError:
            raise yaml.YAMLError(
                f"Unable to load data from {yml_fn}. Check the YAML format.")
    else:
        raise ValueError(f"Invalid value for 'todo': {todo}. Use 'load' or 'dump'.")

    # Manage paths and configurations
    cparser_manager(f"{yml_fn}.yml", savepath, todo=todo, dpath='_saveyaml_',
                    verbose=verbose, config='YAML')

    return data


def get_config_fname_from_varname(
        data, config_fname: Optional[str] = None, config: str = '.yml') -> str:
    """
    Generate a filename based on a variable name for YAML configuration.

    Parameters
    ----------
    data : Any
        The data object from which the variable name will be derived to 
        create a YAML configuration filename.

    config_fname : str, optional
        Custom configuration filename. If `None`, the name of `data` will 
        be used as the filename.

    config : str, default='.yml'
        The file extension/type for the configuration file. Can be '.yml', 
        '.json', or '.csv'.

    Returns
    -------
    str
        A suitable filename for saving the configuration data.

    Raises
    ------
    ValueError
        If `config_fname` cannot be derived or an invalid file type is provided.

    Notes
    -----
    This function supports dynamic filename generation based on variable names,
    which aids in maintaining a clear configuration structure for serialized 
    data. Files are saved with appropriate extensions based on the `config` type.
    """
    # Clean up file extension and validate config type
    config = config.lstrip('.')
    if config_fname is None:
        try:
            config_fname = f"{data}".split('=')[0].strip()
        except Exception as e:
            raise ValueError(f"Unable to determine configuration filename: {str(e)}")
    else:
        config_fname = config_fname.replace(f".{config}", "").replace(".yaml", "")
    
    # Append correct file extension
    return f"{config_fname}.{config}"


def parse_json(
    json_fn: str = None,
    data=None,
    todo: str = 'load',
    savepath: Optional[str] = None,
    verbose: int = 0,
    **jsonkws
):
    """
    Parse and manage JSON configuration files, either loading data from
    or saving data to a JSON file.

    Parameters
    ----------
    json_fn : str, optional
        JSON filename or URL. If `data` is provided and `todo` is 'dump', 
        `json_fn` will be used as the output filename. If `todo` is 'load', 
        `json_fn` is the input filename or URL.

    data : Any, optional
        Data in Python object format to serialize and save if `todo` is 'dump'.

    todo : {'load', 'loads', 'dump', 'dumps'}, default='load'
        Action to perform with JSON:
        - 'load': Load data from a JSON file.
        - 'loads': Parse a JSON string.
        - 'dump': Serialize `data` to a JSON file.
        - 'dumps': Serialize `data` to a JSON string.

    savepath : str, optional
        Path where the JSON file will be saved if `todo` is 'dump'. If 
        `savepath` does not exist, it will save to the default path '_savejson_'.

    verbose : int, default=0
        Controls verbosity of output messages.

    **jsonkws : dict
        Additional keyword arguments passed to `json.dump` or `json.dumps` 
        when saving data.

    Returns
    -------
    Any
        The data loaded from the JSON file or URL if `todo` is 'load', or `data`
        after saving if `todo` is 'dump'.

    Raises
    ------
    json.JSONDecodeError
        If there is an issue with reading or writing the JSON file.

    TypeError
        If the JSON file or data cannot be processed.

    Notes
    -----
    This function uses `json.load`, `json.loads`, `json.dump`, and `json.dumps` 
    for efficient handling of JSON files and strings.

    See Also
    --------
    `fetch_json_data_from_url` : Fetches JSON data from a given URL.
    `get_config_fname_from_varname` : Utility for generating JSON configuration 
    filenames based on variable names.
    """
    # Set task for loading or dumping JSON
    if json_fn and "http" in json_fn:
        todo, json_fn, data = fetch_json_data_from_url(json_fn, todo)

    if 'dump' in todo:
        json_fn = get_config_fname_from_varname(data, json_fn, config='.json')

    JSON = {
        "load": json.load,
        "loads": json.loads,
        "dump": json.dump,
        "dumps": json.dumps
    }

    try:
        if todo == 'load':
            with open(json_fn, "r") as fj:
                data = JSON[todo](fj)
        elif todo == 'loads':
            data = JSON[todo](json_fn)
        elif todo == 'dump':
            with open(f"{json_fn}.json", "w") as fw:
                JSON[todo](data, fw, **jsonkws)
        elif todo == 'dumps':
            data = JSON[todo](data, **jsonkws)
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Unable to {todo} JSON file {json_fn}. Please verify your file.", f'{json_fn!r}', 1)
    except Exception:
        raise TypeError(
            f"Error with {json_fn!r}. Verify your {'file' if 'load' in todo else 'data'}.")

    cparser_manager(
        f"{json_fn}.json", savepath, todo=todo,
        dpath='_savejson_', verbose=verbose, config='JSON'
    )

    return data

def fetch_json_data_from_url(url: str, todo: str = 'load'):
    """
    Retrieve and parse JSON data from a URL.

    Parameters
    ----------
    url : str
        Universal Resource Locator (URL) from which JSON data is fetched.

    todo : {'load', 'dump'}, default='load'
        Action to perform with JSON:
        - 'load': Load JSON data from the URL.
        - 'dump': Parse and prepare data from the URL for saving in a JSON file.

    Returns
    -------
    tuple
        A tuple of `todo` action, filename (or data source), and parsed data.

    Raises
    ------
    urllib.error.URLError
        If there is an issue accessing the URL.

    Notes
    -----
    The function uses `json.loads` to parse data directly from a URL response,
    supporting convenient access to web-hosted JSON content.
    """
    with urllib.request.urlopen(url) as jresponse:
        source = jresponse.read()
    data = json.loads(source)

    if 'load' in todo:
        todo, json_fn = 'loads', source
    elif 'dump' in todo:
        todo, json_fn = 'dumps', '_urlsourcejsonf.json'

    return todo, json_fn, data

   
