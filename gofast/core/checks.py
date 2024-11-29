# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility functions for data validation, assertion, and feature extraction.
Includes checks for dimensionality, type consistency, and feature existence.
Also supports regex-based searches and classification task validation.
"""
from __future__ import print_function
import re
import numbers 
import warnings
import itertools

from typing import Any,  Union,List, Tuple, Optional
import numpy as np
import pandas as pd

from ..api.types import Series, Iterable, _F,  DataFrame 

__all__= [ 
    'assert_ratio',
    'check_dimensionality',
    'check_uniform_type',
    'exist_features',
    'features_in',
    'find_features_in',
    'validate_feature',
    'validate_noise',
    'validate_ratio',
    'is_classification_task',
    'is_depth_in',
    'is_in_if',
    'find_by_regex',
    'find_closest',
    'str2columns',
    'random_state_validator', 
    ]

   
def find_closest(arr, values):
    """
    Find the closest values in an array from a set of target values.

    This function takes an array and a set of target values, and for each 
    target value, finds the closest value in the array. It can handle 
    both scalar and array-like inputs for `values`, ensuring flexibility 
    in usage. The result is either a single closest value or an array 
    of closest values corresponding to each target.

    Parameters
    ----------
    arr : array-like
        The array to search within. It can be a list, tuple, or numpy array 
        of numeric types. If the array is multi-dimensional, it will be 
        flattened to a 1D array.
        
    values : float or array-like
        The target value(s) to find the closest match for in `arr`. This can 
        be a single float or an array of floats.

    Returns
    -------
    numpy.ndarray
        An array of the closest values in `arr` for each target in `values`.
        If `values` is a single float, the function returns a single-element
        array.

    Notes
    -----
    - This function operates by calculating the absolute difference between
      each element in `arr` and each target in `values`, selecting the 
      element with the smallest difference.
    - The function assumes `arr` and `values` contain numeric values, and it
      raises a `TypeError` if they contain non-numeric data.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.checks import find_closest
    >>> find_closest([2, 3, 4, 5], 2.6)
    array([3.])

    >>> find_closest(np.array([[2, 3], [4, 5]]), (2.6, 5.6))
    array([3., 5.])

    See Also
    --------
    numpy.argmin : Find the indices of the minimum values along an axis.
    numpy.abs : Compute the absolute value element-wise.

    References
    ----------
    .. [1] Harris, C. R., et al. "Array programming with NumPy." 
       Nature 585.7825 (2020): 357-362.
    """
    from .validator import _is_numeric_dtype
    arr = is_iterable(arr, exclude_string=True, transform=True)
    values = is_iterable(values, exclude_string=True, transform=True)

    # Validate numeric types in arr and values
    for var, name in zip([arr, values], ['array', 'values']):
        if not _is_numeric_dtype(var, to_array=True):
            raise TypeError(f"Non-numeric data found in {name}.")

    # Convert arr and values to numpy arrays for vectorized operations
    arr = np.array(arr, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    # Flatten arr if it is multi-dimensional
    arr = arr.ravel() if arr.ndim != 1 else arr

    # Find the closest value for each target in values
    closest_values = np.array([
        arr[np.abs(arr - target).argmin()] for target in values
    ])

    return closest_values


def find_by_regex (o , pattern,  func = re.match, **kws ):
    """ Find pattern in object whatever an "iterable" or not. 
    
    when we talk about iterable, a string value is not included.
    
    Parameters 
    -------------
    o: str or iterable,  
        text litteral or an iterable object containing or not the specific 
        object to match. 
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
    
    func: re callable , default=re.match
        regular expression search function. Can be
        [re.match, re.findall, re.search ],or any other regular expression 
        function. 
        
        * ``re.match()``:  function  searches the regular expression pattern and 
            return the first occurrence. The Python RegEx Match method checks 
            for a match only at the beginning of the string. So, if a match is 
            found in the first line, it returns the match object. But if a match 
            is found in some other line, the Python RegEx Match function returns 
            null.
        * ``re.search()``: function will search the regular expression pattern 
            and return the first occurrence. Unlike Python re.match(), it will 
            check all lines of the input string. The Python re.search() function 
            returns a match object when the pattern is found and “null” if 
            the pattern is not found
        * ``re.findall()`` module is used to search for 'all' occurrences that 
            match a given pattern. In contrast, search() module will only 
            return the first occurrence that matches the specified pattern. 
            findall() will iterate over all the lines of the file and will 
            return all non-overlapping matches of pattern in a single step.
    kws: dict, 
        Additional keywords arguments passed to functions :func:`re.match` or 
        :func:`re.search` or :func:`re.findall`. 
        
    Returns 
    -------
    om: list 
        matched object put is the list 
        
    Example
    --------
    >>> from gofast.core.checks import find_by_regex
    >>> from gofast.datasets import load_hlogs 
    >>> X0, _= load_hlogs (as_frame =True )
    >>> columns = X0.columns 
    >>> str_columns =','.join (columns) 
    >>> find_by_regex (str_columns , pattern='depth', func=re.search)
    ... ['depth']
    >>> find_by_regex(columns, pattern ='depth', func=re.search)
    ... ['depth_top', 'depth_bottom']
    
    """
    om = [] 
    if isinstance (o, str): 
        om = func ( pattern=pattern , string = o, **kws)
        if om: 
            om= om.group() 
        om =[om]
    elif is_iterable(o): 
        o = list(o) 
        for s in o : 
            z = func (pattern =pattern , string = s, **kws)
            if z : 
                om.append (s) 
                
    if func.__name__=='findall': 
        om = list(itertools.chain (*om )) 
    # keep None is nothing 
    # fit the corresponding pattern 
    if len(om) ==0 or om[0] is None: 
        om = None 
    return  om 
    
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
        >>> from gofast.core.checks import is_in_if 
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
    elif not is_iterable(o): 
        raise TypeError (f"Expect an iterable object, not {type(o).__name__!r}")
    # find intersect object 
    s= set (o).intersection (items) 
    
    miss_items = list(s.difference (o)) if len(s) > len(
        items) else list(set(items).difference (s)) 

    if return_diff or return_intersect: 
        error ='ignore'
    
    if len(miss_items)!=0 :
        if error =='raise': 
            v= _smart_format(miss_items)
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
  

def is_depth_in (X, name, columns = None, error= 'ignore'): 
    """ Assert wether depth exists in the data from column attributes.  
    
    If name is an integer value, it assumes to be the index in the columns 
    of the dataframe if not exist , a warming will be show to user. 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
        
    :param columns: list,
        New labels to replace the columns in the dataframe. If given , it 
        should fit the number of colums of `X`. 
        
    :param name: str, int  
        depth name in the dataframe or index to retreive the name of the depth 
        in dataframe 
    :param error: str , default='ignore'
        Raise or ignore when depth is not found in the dataframe. Whe error is 
        set to ``ignore``, a pseudo-depth is created using the lenght of the 
        the dataframe, otherwise a valueError raises.
        
    :return: X, depth 
        Dataframe without the depth columns and depth values.
    """
    
    X= _assert_all_types( X, pd.DataFrame )
    if columns is not None: 
        columns = list(columns)
        if not is_iterable(columns): 
            raise TypeError("columns expects an iterable object."
                            f" got {type (columns).__name__!r}")
        if len(columns ) != len(X.columns): 
            warnings.warn("Cannot rename columns with new labels. Expect "
                          "a size to be consistent with the columns X."
                          f" {len(columns)} and {len(X.columns)} are given."
                          )
        else : 
            X.columns = columns # rename columns
        
    else:  columns = list(X.columns) 
    
    _assert_all_types(name,str, int, float )
    
    # if name is given as indices 
    # collect the name at that index 
    if isinstance (name, (int, float) )  :     
        name = int (name )
        if name > len(columns): 
            warnings.warn ("Name index {name} is out of the columns range."
                           f" Max index of columns is {len(columns)}")
            name = None 
        else : 
            name = columns.pop (name)
    
    elif isinstance (name, str): 
        # find in columns whether a name can be 
        # found. Note that all name does not need 
        # to be written completely 
        # for instance name =depth can retrieved 
        # ['depth_top, 'depth_bottom'] , in that case 
        # the first occurence is selected i.e. 'depth_top'
        n = find_by_regex( 
            columns, pattern=fr'{name}', func=re.search)

        if n is not None:
            name = n[0]
            
        # for consistency , recheck all and let 
        # a warning to user 
        if name not in columns :
            msg = f"Name {name!r} does not match any column names."
            if error =='raise': 
                raise ValueError (msg)

            warnings.warn(msg)
            name =None  
            
    # now create a pseudo-depth 
    # as a range of len X 
    if name is None: 
        if error =='raise':
            raise ValueError ("Depth column not found in dataframe."
                              )
        depth = pd.Series ( np.arange ( len(X)), name ='depth (m)') 
    else : 
        # if depth name exists, 
        # remove it from X  
        depth = X.pop (name ) 
        
    return  X , depth     
    
def is_classification_task(
    *y, max_unique_values=10
    ):
    """
    Check whether the given arrays are for a classification task.

    This function assumes that if all values in the provided arrays are 
    integers and the number of unique values is within the specified
    threshold, it is a classification task.

    Parameters
    ----------
    *y : list or numpy.array
        A variable number of arrays representing actual values, 
        predicted values, etc.
    max_unique_values : int, optional
        The maximum number of unique values to consider the task 
        as classification. 
        Default is 10.

    Returns
    -------
    bool
        True if the provided arrays are for a classification task, 
        False otherwise.

    Examples
    --------
    >>> from gofast.core.checks import is_classification_task 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> is_classification_task(y_true, y_pred)
    True
    """
    max_unique_values = int (
        _assert_all_types(max_unique_values, 
                          int, float, objname="Max Unique values")
                             )
    # Combine all arrays for analysis
    combined = np.concatenate(y)

    # Check if all elements are integers
    if ( 
            not all(isinstance(x, int) for x in combined) 
            and not combined.dtype.kind in 'iu'
            ):
        return False

    # Check the number of unique elements
    unique_values = np.unique(combined)
    # check Arbitrary threshold for number of classes
    if len(unique_values) > max_unique_values:
        return False

    return True

def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise


def validate_feature(data: Union[DataFrame, Series],  features: List[str],
                  verbose: str = 'raise') -> bool:
 """
 Validate the existence of specified features in a DataFrame or Series.

 Parameters
 ----------
 data : DataFrame or Series
     The DataFrame or Series to validate feature existence.
 features : list of str
     List of features to check for existence in the data.
 verbose : str, {'raise', 'ignore'}, optional
     Specify how to handle the absence of features. 'raise' (default) will raise
     a ValueError if any feature is missing, while 'ignore' will return a
     boolean indicating whether all features exist.

 Returns
 -------
 bool
     True if all specified features exist in the data, False otherwise.

 Examples
 --------
 >>> from gofast.core.checks import validate_feature
 >>> import pandas as pd
 >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
 >>> result = validate_feature(data, ['A', 'C'], verbose='raise')
 >>> print(result)  # This will raise a ValueError
 """
 if isinstance(data, pd.Series):
     data = data.to_frame().T  # Convert Series to DataFrame
 features= is_iterable(features, exclude_string= True, transform =True )
 present_features = set(features).intersection(data.columns)

 if len(present_features) != len(features):
     missing_features = set(features).difference(present_features)
     verb =" is" if len(missing_features) <2 else "s are"
     if verbose == 'raise':
         raise ValueError(f"The following feature{verb} missing in the "
                          f"data: {_smart_format(missing_features)}.")
     return False

 return True

def features_in(
    *data: Union[pd.DataFrame, pd.Series], features: List[str],
    error: str = 'ignore') -> List[bool]:
    """
    Control whether the specified features exist in multiple datasets.

    Parameters
    ----------
    *data : DataFrame or Series arguments
        Multiple DataFrames or Series to check for feature existence.
    features : list of str
        List of features to check for existence in the datasets.
    error : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'ignore' (default) will ignore
        a ValueError for each dataset with missing features, while 'ignore' will
        return a list of booleans indicating whether all features exist in each dataset.

    Returns
    -------
    list of bool
        A list of booleans indicating whether the specified features exist in each dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import features_in
    >>> data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> data2 = pd.Series([5, 6], name='C')
    >>> data3 = pd.DataFrame({'X': [7, 8]})
    >>> features = ['A', 'C']
    >>> results1 = features_in(data1, data2, features, error='raise')
    >>> print(results1)  # This will raise a ValueError for the first dataset
    >>> results2 = features_in(data1, data3, features, error='ignore')
    >>> print(results2)  # This will return [True, False]
    """
    results = []

    for dataset in data:
        results.append(validate_feature(dataset, features, verbose=error))

    return results

def find_features_in(
    data: DataFrame = None,
    features: List[str] = None,
    parse_features: bool = False,
    return_frames: bool = False,
) -> Tuple[Union[List[str], DataFrame], Union[List[str], DataFrame]]:
    """
    Retrieve the categorical or numerical features from the dataset.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with columns representing the features.
    features : list of str, optional
        List of column names. If provided, the DataFrame will be restricted
        to only include the specified features before searching for numerical
        and categorical features. An error will be raised if any specified
        feature is missing in the DataFrame.
    return_frames : bool, optional
        If True, it returns two separate DataFrames (cat & num). Otherwise, it
        returns only the column names of categorical and numerical features.
    parse_features : bool, default False
        Use default parsers to parse string items into an iterable object.

    Returns
    -------
    Tuple : List[str] or DataFrame
        The names or DataFrames of categorical and numerical features.

    Examples
    --------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.tools.mlutils import find_features_in
    >>> data = fetch_data('bagoue').frame 
    >>> cat, num = find_features_in(data)
    >>> cat, num
    ... (['type', 'geol', 'shape', 'name', 'flow'],
    ...  ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(data, features=['geol', 'ohmS', 'sfi'])
    >>> cat, num
    ... (['geol'], ['ohmS', 'sfi'])
    """
    from .array_manager import to_numeric_dtypes 
    
    if not isinstance (data, pd.DataFrame):
        raise TypeError(f"Expect a DataFrame. Got {type(data).__name__!r}")

    if features is not None:
        features = list(
            is_iterable(
                features,
                exclude_string=True,
                transform=True,
                parse_string=parse_features,
            )
        )

    if features is None:
        features = list(data.columns)

    validate_feature(data, list(features))
    data = data[features].copy()

    # Get numerical features
    data, numnames, catnames = to_numeric_dtypes(
        data, return_feature_types=True )

    if catnames is None:
        catnames = []

    return (data[catnames], data[numnames]) if return_frames else (
        list(catnames), list(numnames)
    )



def check_uniform_type(
    values: Union[Iterable[Any], Any],
    items_to_compare: Union[Iterable[Any], Any] = None,
    raise_exception: bool = True,
    convert_values: bool = False,
    return_types: bool = False,
    target_type: type = None,
    allow_mismatch: bool = True,
    infer_types: bool = False,
    comparison_method: str = 'intersection',
    custom_conversion_func: _F[Any] = None,
    return_func: bool = False
) -> Union[bool, List[type], Tuple[Iterable[Any], List[type]], _F]:
    """
    Checks whether elements in `values` are of uniform type. 
    
    Optionally comparing them against another set of items or converting all 
    values to a target type. Can return a callable for deferred execution of 
    the specified logic.Function is useful for validating data uniformity, 
    especially before performing operations that assume homogeneity of the 
    input types.
    

    Parameters
    ----------
    values : Iterable[Any] or Any
        An iterable containing items to check. If a non-iterable item is provided,
        it is treated as a single-element iterable.
    items_to_compare : Iterable[Any] or Any, optional
        An iterable of items to compare against `values`. If specified, the
        `comparison_method` is used to perform the comparison.
    raise_exception : bool, default True
        If True, raises an exception when a uniform type is not found or other
        constraints are not met. Otherwise, issues a warning.
    convert_values : bool, default False
        If True, tries to convert all `values` to `target_type`. Requires
        `target_type` to be specified.
    return_types : bool, default False
        If True, returns the types of the items in `values`.
    target_type : type, optional
        The target type to which `values` should be converted if `convert_values`
        is True.
    allow_mismatch : bool, default True
        If False, requires all values to be of identical types; otherwise,
        allows type mismatch.
    infer_types : bool, default False
        If True and different types are found, returns the types of each item
        in `values` in order.
    comparison_method : str, default 'intersection'
        The method used to compare `values` against `items_to_compare`. Must
        be one of the set comparison methods ('difference', 'intersection', etc.).
    custom_conversion_func : Callable[[Any], Any], optional
        A custom function for converting items in `values` to another type.
    return_func : bool, default False
        If True, returns a callable that encapsulates the logic based on the 
        other parameters.

    Returns
    -------
    Union[bool, List[type], Tuple[Iterable[Any], List[type]], Callable]
        The result based on the specified parameters. This can be: 
        - A boolean indicating whether all values are of the same type.
        - The common type of all values if `return_types` is True.
        - A tuple containing the converted values and their types if `convert_values`
          and `return_types` are both True.
        - a callable encapsulating the specified logic for deferred execution.
        
    Examples
    --------
    >>> from gofast.core.checks import check_uniform_type
    >>> check_uniform_type([1, 2, 3])
    True

    >>> check_uniform_type([1, '2', 3], allow_mismatch=False, raise_exception=False)
    False

    >>> deferred_check = check_uniform_type([1, 2, '3'], convert_values=True, 
    ...                                        target_type=int, return_func=True)
    >>> deferred_check()
    [1, 2, 3]

    Notes
    -----
    The function is designed to be flexible, supporting immediate or deferred execution,
    with options for type conversion and detailed type information retrieval.
    """
    def operation():
        # Convert values and items_to_compare to lists if 
        # they're not already iterable
        if isinstance(values, Iterable) and not isinstance(values, str):
            val_list = list(values)
        else:
            val_list = [values]

        if items_to_compare is not None:
            if isinstance(items_to_compare, Iterable) and not isinstance(
                    items_to_compare, str):
                comp_list = list(items_to_compare)
            else:
                comp_list = [items_to_compare]
        else:
            comp_list = []

        # Extract types
        val_types = set(type(v) for v in val_list)
        comp_types = set(type(c) for c in comp_list) if comp_list else set()

        # Compare types
        if comparison_method == 'intersection':
            common_types = val_types.intersection(comp_types) if comp_types else val_types
        elif comparison_method == 'difference':
            common_types = val_types.difference(comp_types)
        else:
            if raise_exception:
                raise ValueError(f"Invalid comparison method: {comparison_method}")
            return False

        # Check for type uniformity
        if not allow_mismatch and len(common_types) > 1:
            if raise_exception:
                raise ValueError("Not all values are the same type.")
            return False

        # Conversion
        if convert_values:
            if not target_type and not custom_conversion_func:
                if raise_exception:
                    raise ValueError("Target type or custom conversion "
                                     "function must be specified for conversion.")
                return False
            try:
                if custom_conversion_func:
                    converted_values = [custom_conversion_func(v) for v in val_list]
                else:
                    converted_values = [target_type(v) for v in val_list]
            except Exception as e:
                if raise_exception:
                    raise ValueError(f"Conversion failed: {e}")
                return False
            if return_types:
                return converted_values, [type(v) for v in converted_values]
            return converted_values

        # Return types
        if return_types:
            if infer_types or len(common_types) > 1:
                return [type(v) for v in val_list]
            return list(common_types)

        return True

    return operation if return_func else operation()


def check_dimensionality(obj, data, z, x):
 """ Check dimensionality of data and fix it.
 
 :param obj: Object, can be a class logged or else.
 :param data: 2D grid data of ndarray (z, x) dimensions.
 :param z: array-like should be reduced along the row axis.
 :param x: arraylike should be reduced along the columns axis.
 
 """
 def reduce_shape(Xshape, x, axis_name=None): 
     """ Reduce shape to keep the same shape"""
     mess ="`{0}` shape({1}) {2} than the data shape `{0}` = ({3})."
     ox = len(x) 
     dsh = Xshape 
     if len(x) > Xshape : 
         x = x[: int (Xshape)]
         obj._logging.debug(''.join([
             f"Resize {axis_name!r}={ox!r} to {Xshape!r}.", 
             mess.format(axis_name, len(x),'more',Xshape)])) 
                                 
     elif len(x) < Xshape: 
         Xshape = len(x)
         obj._logging.debug(''.join([
             f"Resize {axis_name!r}={dsh!r} to {Xshape!r}.",
             mess.format(axis_name, len(x),'less', Xshape)]))
     return int(Xshape), x 
 
 sz0, z = reduce_shape(data.shape[0], 
                       x=z, axis_name ='Z')
 sx0, x =reduce_shape (data.shape[1],
                       x=x, axis_name ='X')
 data = data [:sz0, :sx0]
 
 return data , z, x 


def assert_ratio(
    v: Union[str, float, int],
    bounds: Optional[Tuple[float, float]] = None,
    exclude_values: Optional[Union[float, List[float]]] = None,
    in_percent: bool = False,
    inclusive: bool = True,
    name: str = 'ratio'
) -> float:
    """
    Asserts that a given value falls within a specified range and does not
    match any excluded values. Optionally converts the value to a percentage.
    
    This function is useful for validating ratio or rate values in data 
    preprocessing, ensuring they meet defined criteria before further 
    analysis or modeling.
    
    Parameters
    ----------
    v : Union[str, float, int]
        The ratio value to assert. Can be a string (possibly containing 
        a percentage sign), float, or integer.
        
    bounds : Optional[Tuple[float, float]], default=None
        A tuple specifying the lower and upper bounds (inclusive by default) 
        within which the value `v` must lie. If `None`, no bounds are enforced.
        
    exclude_values : Optional[Union[float, List[float]]], default=None
        Specific value(s) that `v` must not equal. Can be a single float or a 
        list of floats. If provided, `v` is checked against these excluded 
        values after any necessary conversions.
        
    in_percent : bool, default=False
        If `True`, interprets the input value `v` as a percentage and converts 
        it to its decimal form (e.g., 50 becomes 0.5). If `v` is a string 
        containing a `%` sign, it is automatically converted to decimal.
        
    inclusive : bool, default=True
        Determines whether the bounds are inclusive. If `True`, `v` can be equal 
        to the lower and upper bounds. If `False`, `v` must be strictly 
        greater than the lower bound and strictly less than the upper bound.
        
    name : str, default='ratio'
        The descriptive name of the value being asserted. This is used in error 
        messages for clarity.
    
    Returns
    -------
    float
        The validated (and possibly converted) ratio value.
        
    Raises
    ------
    TypeError
        If `v` cannot be converted to a float.
        
    ValueError
        If `v` is outside the specified bounds or matches any excluded values.
    
    Examples
    --------
    1. **Basic Usage with Bounds:**
    
        ```python
        from gofast.core.checks import assert_ratio
        assert_ratio(0.5, bounds=(0.0, 1.0))
        # Returns: 0.5
        ```
    
    2. **String Input with Percentage:**
    
        ```python
        assert_ratio("75%", in_percent=True)
        # Returns: 0.75
        ```
    
    3. **Excluding Specific Values:**
    
        ```python
        assert_ratio(0.5, bounds=(0.0, 1.0), exclude_values=0.5)
        # Raises ValueError
        ```
    
    4. **Multiple Excluded Values and Exclusive Bounds:**
    
        ```python
        assert_ratio(0.3, bounds=(0.0, 1.0), exclude_values=[0.2, 0.4], inclusive=False)
        # Returns: 0.3
        ```
    
    Notes
    -----
    - The function first attempts to convert the input `v` to a float. 
      If `in_percent` is `True`, it converts percentage values to 
      their decimal equivalents.
    - Bounds can be set to define a valid range for `v`. If `inclusive` 
      is set to `False`, the bounds are treated as exclusive.
    - Excluded values are checked after any necessary conversions.
    - If `exclude_values` is provided without specifying `bounds`, the 
      function will only check for excluded values.
    
    References
    ----------
    - [Python `float()` Function](https://docs.python.org/3/library/functions.html#float)
    - [Warnings in Python](https://docs.python.org/3/library/warnings.html)
    """
    
    # Initialize exclusion list
    if exclude_values is not None and not isinstance(exclude_values, list):
        exclude_values = [exclude_values]
    
    # Regular expression to detect percentage in strings
    percent_pattern = re.compile(r'^\s*[-+]?\d+(\.\d+)?%\s*$')
    
    # Check and convert string inputs
    if isinstance(v, str):
        v = v.strip()
        if percent_pattern.match(v):
            in_percent = True
            v = v.replace('%', '').strip()
    
    try:
        # Convert to float
        v = float(v)
    except (TypeError, ValueError):
        raise TypeError(
            f"Unable to convert {type(v).__name__!r} value '{v}' to float."
        )
    
    # Convert to percentage if required
    if in_percent:
        if 0 <= v <= 100:
            v /= 100.0
        elif 0 <= v <= 1:
            warnings.warn(
                f"The value {v} seems already in decimal form; "
                f"no conversion applied for {name}.",
                UserWarning
            )
        else:
            raise ValueError(
                f"When 'in_percent' is True, {name} should be between "
                f"0 and 100, got {v * 100 if 0 <= v <=1 else v}."
            )
    
    # Check bounds if specified
    if bounds:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                 "'bounds' must be a tuple of two"
                f" floats, got {type(bounds).__name__}."
            )
        lower, upper = bounds
        if inclusive:
            if not (lower <= v <= upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower}"
                    f" and {upper} inclusive, got {v}."
                )
        else:
            if not (lower < v < upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower} and "
                    f"{upper} exclusive, got {v}."
                )
    
    # Check excluded values
    if exclude_values:
        if v in exclude_values:
            if len(exclude_values) == 1:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be {exclude_values[0]}."
                    )
            else:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be one of {exclude_values}."
                    )
            raise ValueError(exclusion_msg)
    
    return v


def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value


def exist_features(
    df: pd.DataFrame, 
    features, 
    error='raise',  
    name="Feature"
) -> bool:
    """
    Check whether the specified features exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be checked.
    features : list or str
        List of feature names (str) to check for in the dataframe. 
        If a string is provided, it will be treated as a list with 
        a single feature.
    error : str, optional, default 'raise'
        Action to take if features are not found. Can be one of:
        - 'raise' (default): Raise a ValueError.
        - 'warn': Issue a warning and return False.
        - 'ignore': Do nothing if features are not found.
    name : str, optional, default 'Feature'
        Name of the feature(s) being checked (default is 'Feature').

    Returns
    -------
    bool
        Returns True if all features exist in the dataframe, otherwise False.

    Raises
    ------
    ValueError
        If 'error' is 'raise' and features are not found.
    
    Warns
    -----
    UserWarning
        If 'error' is 'warn' and features are missing.

    Notes
    -----
    This function ensures that all the specified features exist in the
    dataframe. If the 'error' parameter is set to 'warn', the function 
    will issue a warning instead of raising an error when a feature 
    is missing, and return False.

    References
    ----------
    - pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from gofast.core.checks import exist_features
    >>> import pandas as pd

    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })

    >>> # Check for missing features with 'raise' error
    >>> exist_features(df, ['feature1', 'feature4'], error='raise')
    Traceback (most recent call last):
        ...
    ValueError: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'warn' error
    >>> exist_features(df, ['feature1', 'feature4'], error='warn')
    UserWarning: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'ignore' error
    >>> exist_features(df, ['feature1', 'feature4'], error='ignore')
    False
    """
    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")

    # Normalize the error parameter to lowercase and strip whitespace
    error = error.lower().strip()

    # Validate the 'error' parameter
    if error not in ['raise', 'ignore', 'warn']:
        raise ValueError(
            "Invalid value for 'error'. Expected"
            " one of ['raise', 'ignore', 'warn'].")

    # Ensure 'features' is a list-like structure
    if isinstance(features, str):
        features = [features]

    # Validate that 'features' is one of the allowed types
    features = _assert_all_types(features, (list, tuple, np.ndarray))

    # Get the intersection of features with the dataframe columns
    existing_features = set(features).intersection(df.columns)

    # If all features exist, return True
    if len(existing_features) == len(features):
        return True

    # Calculate the missing features
    missing_features = set(features) - existing_features

    # If there are missing features, handle according to 'error' type
    if missing_features:
        msg = f"{name}{'s' if len(features) > 1 else ''}"

        if error == 'raise':
            raise ValueError(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe."
            )

        elif error == 'warn':
            warnings.warn(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe.",
                UserWarning
            )
            return False

        # If 'error' is 'ignore', simply return False
        return False

    return True

def is_iterable (
        y, exclude_string= False, transform = False , parse_string =False, 
)->Union [bool , list]: 
    """ Asserts iterable object and returns boolean or transform object into
     an iterable.
    
    Function can also transform a non-iterable object to an iterable if 
    `transform` is set to ``True``.
    
    :param y: any, object to be asserted 
    :param exclude_string: bool, does not consider string as an iterable 
        object if `y` is passed as a string object. 
    :param transform: bool, transform  `y` to an iterable objects. But default 
        puts `y` in a list object. 
    :param parse_string: bool, parse string and convert the list of string 
        into iterable object is the `y` is a string object and containg the 
        word separator character '[#&.*@!_,;\s-]'. Refer to the function 
        :func:`~gofast.core.checks.str2columns` documentation.
        
    :returns: 
        - bool, or iterable object if `transform` is set to ``True``. 
        
    .. note:: 
        Parameter `parse_string` expects `transform` to be ``True``, otherwise 
        a ValueError will raise. Note :func:`.is_iterable` is not dedicated 
        for string parsing. It parses string using the default behaviour of 
        :func:`.str2columns`. Use the latter for string parsing instead. 
        
    :Examples: 
    >>> from gofast.coreutils.is_iterable 
    >>> is_iterable ('iterable', exclude_string= True ) 
    Out[28]: False
    >>> is_iterable ('iterable', exclude_string= True , transform =True)
    Out[29]: ['iterable']
    >>> is_iterable ('iterable', transform =True)
    Out[30]: 'iterable'
    >>> is_iterable ('iterable', transform =True, parse_string=True)
    Out[31]: ['iterable']
    >>> is_iterable ('iterable', transform =True, exclude_string =True, 
                     parse_string=True)
    Out[32]: ['iterable']
    >>> is_iterable ('parse iterable object', parse_string=True, 
                     transform =True)
    Out[40]: ['parse', 'iterable', 'object']
    """
    if (parse_string and not transform) and isinstance (y, str): 
        raise ValueError ("Cannot parse the given string. Set 'transform' to"
                          " ``True`` otherwise use the 'str2columns' utils"
                          " from 'gofast.core.checks' instead.")
    y = str2columns(y) if isinstance(y, str) and parse_string else y 
    
    isiter = False  if exclude_string and isinstance (
        y, str) else hasattr (y, '__iter__')
    
    return ( y if isiter else [ y ] )  if transform else isiter 

def _smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable object.
    """
    str_litteral =''
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" {choice} {iter_obj[-1]!r}"
    return str_litteral

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
    >>> from gofast.core.checks import str2columns 
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

def _assert_all_types(
    obj: object,
    *expected_objtype: type,
    objname: str = None,
) -> object:
    """
    Quick assertion to check if an object is of an expected type.

    Parameters
    ----------
    obj : object
        The object whose type is being checked.
    expected_objtype : type
        One or more types to check against. If the object's type
        does not match any of the provided types, a TypeError is raised.
    objname : str, optional
        The name of the object being checked, used to customize the
        error message. If not provided, a generic message is used.

    Raises
    ------
    TypeError
        If the object's type does not match any of the expected types.

    Returns
    -------
    object
        The original object if its type matches one of the expected types.

    Notes
    -----
    This function raises a `TypeError` if the object's type does not
    match the expected type(s). The error message can be customized by
    providing the `objname` argument.
    """
    # if np.issubdtype(a1.dtype, np.integer): 
    if not isinstance(obj, expected_objtype):
        n = str(objname) + ' expects' if objname is not None else 'Expects'
        raise TypeError(
            f"{n} type{'s' if len(expected_objtype) > 1 else ''} "
            f"{_smart_format(tuple(o.__name__ for o in expected_objtype))} "
            f"but {type(obj).__name__!r} is given."
        )

    return obj

def random_state_validator(seed):
    """Turn seed into a Numpy-Random-RandomState instance.
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
