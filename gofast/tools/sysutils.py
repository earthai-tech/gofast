# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
System utilities module for managing color handling, regex-based searches, 
projections, and system-level operations.

This module provides utilities essential for system-level tasks such as
color management, regular expression searching, and projection validation,
along with other miscellaneous system operations.
"""

import re 
import inspect 
import itertools 
from typing import Union, Tuple, Dict,Optional, List
from typing import Sequence, Any
import multiprocessing
from concurrent.futures import as_completed 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np 

from ..api.types import _F 
from .coreutils import is_iterable, _assert_all_types 
from ._dependency import import_optional_dependency 

__all__ = ["parallelize_jobs","generic_getattr", "smart_strobj_recognition", 
           "find_by_regex", "repr_callable_obj"]

def parallelize_jobs(
    function: _F,
    tasks: Sequence[Dict[str, Any]] = (),
    n_jobs: Optional[int] = None,
    executor_type: str = 'process') -> list:
    """
    Parallelize the execution of a callable across multiple processors, 
    supporting both positional and keyword arguments.

    Parameters
    ----------
    function : Callable[..., Any]
        The function to execute in parallel. This function must be picklable 
        if using `executor_type='process'`.
    tasks : Sequence[Dict[str, Any]], optional
        A sequence of dictionaries, where each dictionary contains 
        two keys: 'args' (a tuple) for positional arguments,
        and 'kwargs' (a dict) for keyword arguments, for one execution of
        `function`. Defaults to an empty sequence.
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel. `None` or `1` uses a single 
        processor, any positive integer specifies the
        exact number of processors to use, `-1` uses all available processors. 
        Default is None (1 processor).
    executor_type : str, optional
        The type of executor to use. Can be 'process' for CPU-bound tasks or
        'thread' for I/O-bound tasks. Default is 'process'.

    Returns
    -------
    list
        A list of results from the function executions.

    Raises
    ------
    ValueError
        If `function` is not picklable when using 'process' as `executor_type`.

    Examples
    --------
    >>> from gofast.tools.coreutils import parallelize_jobs
    >>> def greet(name, greeting='Hello'):
    ...     return f"{greeting}, {name}!"
    >>> tasks = [
    ...     {'args': ('John',), 'kwargs': {'greeting': 'Hi'}},
    ...     {'args': ('Jane',), 'kwargs': {}}
    ... ]
    >>> results = parallelize_jobs(greet, tasks, n_jobs=2)
    >>> print(results)
    ['Hi, John!', 'Hello, Jane!']
    """
    if executor_type == 'process':
        import_optional_dependency("cloudpickle")
        import cloudpickle
        try:
            cloudpickle.dumps(function)
        except cloudpickle.PicklingError:
            raise ValueError("The function to be parallelized must be "
                             "picklable when using 'process' executor.")

    num_workers = multiprocessing.cpu_count() if n_jobs == -1 else (
        1 if n_jobs is None else n_jobs)
    
    ExecutorClass = ProcessPoolExecutor if executor_type == 'process' \
        else ThreadPoolExecutor
    
    results = []
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = [executor.submit(function, *task.get('args', ()),
                                   **task.get('kwargs', {})) for task in tasks]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results
 
    
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
    >>> from gofast.tools.coreutils import find_by_regex
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

def generic_getattr(obj, name, default_value=None):
    """
    A generic attribute accessor for any class instance.

    This function attempts to retrieve an attribute from the given object.
    If the attribute is not found, it provides a meaningful error message.

    Parameters:
    ----------
    obj : object
        The object from which to retrieve the attribute.

    name : str
        The name of the attribute to retrieve.

    default_value : any, optional
        A default value to return if the attribute is not found. If None,
        an AttributeError will be raised.

    Returns:
    -------
    any
        The value of the retrieved attribute or the default value.

    Raises:
    ------
    AttributeError
        If the attribute is not found and no default value is provided.

    Examples:
    --------
    >>> from gofast.tools.coreutils import generic_getattr
    >>> class MyClass:
    >>>     def __init__(self, a, b):
    >>>         self.a = a
    >>>         self.b = b
    >>> obj = MyClass(1, 2)
    >>> print(generic_getattr(obj, 'a'))  # Prints: 1
    >>> print(generic_getattr(obj, 'c', 'default'))  # Prints: 'default'
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    
    if default_value is not None:
        return default_value

    # Attempt to find a similar attribute name for a more informative error
    similar_attr = _find_similar_attribute(obj, name)
    suggestion = f". Did you mean '{similar_attr}'?" if similar_attr else ""

    raise AttributeError(f"'{obj.__class__.__name__}' object has no "
                         f"attribute '{name}'{suggestion}")

def _find_similar_attribute(obj, name):
    """
    Attempts to find a similar attribute name in the object's dictionary.

    Parameters
    ----------
    obj : object
        The object whose attributes are being checked.
    name : str
        The name of the attribute to find a similar match for.

    Returns
    -------
    str or None
        A similar attribute name if found, otherwise None.
    """
    rv = smart_strobj_recognition(name, obj.__dict__, deep =True)
    return rv 
 
def smart_strobj_recognition(
        name: str  ,
        container: Union [List , Tuple , Dict[Any, Any ]],
        stripitems: Union [str , List , Tuple] = '_', 
        deep: bool = False,  
) -> str : 
    """ Find the likelihood word in the whole containers and 
    returns the value.
    
    :param name: str - Value of to search. I can not match the exact word in 
    the `container`
    :param container: list, tuple, dict- container of the many string words. 
    :param stripitems: str - 'str' items values to sanitize the  content 
        element of the dummy containers. if different items are provided, they 
        can be separated by ``:``, ``,`` and ``;``. The items separators 
        aforementioned can not  be used as a component in the `name`. For 
        isntance:: 
            
            name= 'dipole_'; stripitems='_' -> means remove the '_'
            under the ``dipole_``
            name= '+dipole__'; stripitems ='+;__'-> means remove the '+' and
            '__' under the value `name`. 
        
    :param deep: bool - Kind of research. Go deeper by looping each items 
         for find the initials that can fit the name. Note that, if given, 
         the first occurence should be consider as the best name... 
         
    :return: Likelihood object from `container`  or Nonetype if none object is
        detected.
        
    :Example:
        >>> from gofast.tools.coreutils import smart_strobj_recognition
        >>> from gofast.methods import ResistivityProfiling 
        >>> rObj = ResistivityProfiling(AB= 200, MN= 20,)
        >>> smart_strobj_recognition ('dip', robj.__dict__))
        ... None 
        >>> smart_strobj_recognition ('dipole_', robj.__dict__))
        ... dipole 
        >>> smart_strobj_recognition ('dip', robj.__dict__,deep=True )
        ... dipole 
        >>> smart_strobj_recognition (
            '+_dipole___', robj.__dict__,deep=True , stripitems ='+;_')
        ... 'dipole'
        
    """

    stripitems =_assert_all_types(stripitems , str, list, tuple) 
    container = _assert_all_types(container, list, tuple, dict)
    ix , rv = None , None 
    
    if isinstance (stripitems , str): 
        for sep in (':', ",", ";"): # when strip ='a,b,c' seperated object
            if sep in stripitems:
                stripitems = stripitems.strip().split(sep) ; break
        if isinstance(stripitems, str): 
            stripitems =[stripitems]
            
    # sanitize the name. 
    for s in stripitems :
        name = name.strip(s)     
        
    if isinstance(container, dict) : 
        #get only the key values and lower them 
        container_ = list(map (lambda x :x.lower(), container.keys())) 
    else :
        # for consistency put on list if values are in tuple. 
        container_ = list(container)
        
    # sanitize our dummny container item ... 
    #container_ = [it.strip(s) for it in container_ for s in stripitems ]
    if name.lower() in container_: 
        try:
            ix = container_.index (name)
        except ValueError: 
            raise AttributeError(f"{name!r} attribute is not defined")
        
    if deep and ix is None:
        # go deeper in the search... 
        for ii, n in enumerate (container_) : 
            if n.find(name.lower())>=0 : 
                ix =ii ; break 
    
    if ix is not None: 
        if isinstance(container, dict): 
            rv= list(container.keys())[ix] 
        else : rv= container[ix] 

    return  rv 

def repr_callable_obj(obj: _F  , skip = None ): 
    """ Represent callable objects. 
    
    Format class, function and instances objects. 
    
    :param obj: class, func or instances
        object to format. 
    :param skip: str , 
        attribute name that is not end with '_' and whom it needs to be 
        skipped. 
        
    :Raises: TypeError - If object is not a callable or instanciated. 
    
    :Examples: 
        
    >>> from gofast.tools.coreutils import repr_callable_obj
    >>> from gofast.methods.electrical import  ResistivityProfiling
    >>> repr_callable_obj(ResistivityProfiling)
    ... 'ResistivityProfiling(station= None, dipole= 10.0, 
            auto_station= False, kws= None)'
    >>> robj= ResistivityProfiling (AB=200, MN=20, station ='S07')
    >>> repr_callable_obj(robj)
    ... 'ResistivityProfiling(AB= 200, MN= 20, arrangememt= schlumberger, ... ,
        dipole= 10.0, station= S07, auto= False)'
    >>> repr_callable_obj(robj.fit)
    ... 'fit(data= None, kws= None)'
    
    """
    regex = re.compile (r"[{'}]")
    
    # inspect.formatargspec(*inspect.getfullargspec(cls_or_func))
    if not hasattr (obj, '__call__') and not hasattr(obj, '__dict__'): 
        raise TypeError (
            f'Format only callabe objects: Got {type (obj).__name__!r}')
        
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        objname = obj.__name__
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        objname=obj.__class__.__name__
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not ((k.endswith('_') or k.startswith('_') 
                                  # remove the dict objects
                                  or k.endswith('_kws') or k.endswith('_props'))
                                 )
                         }
    if skip is not None : 
        # skip some inner params 
        # remove them as the main function or class params 
        if isinstance(skip, (tuple, list, np.ndarray)): 
            skip = list(map(str, skip ))
            exs = [key for key in PARAMS_VALUES.keys() if key in skip]
        else:
            skip =str(skip).strip() 
            exs = [key for key in PARAMS_VALUES.keys() if key.find(skip)>=0]
 
        for d in exs: 
            PARAMS_VALUES.pop(d, None) 
            
    # use ellipsis as internal to stdout more than seven params items 
    if len(PARAMS_VALUES) >= 7 : 
        f = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[:3]}
        e = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[-3:]}
        
        PARAMS_VALUES= str(f) + ' ... ' + str(e )

    return str(objname) + '(' + regex.sub('', str (PARAMS_VALUES)
                                          ).replace(':', '=') +')'