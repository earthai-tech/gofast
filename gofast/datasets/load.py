# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio 

"""
load different data as a function 
=================================

Inspired from the machine learning popular dataset loading 
"""
import os
import scipy 
import joblib
import numpy as np
from importlib import resources 
from importlib.resources import files
import pandas as pd 

from ..tools.baseutils import read_data, fancier_downloader, check_file_exists   
from ..tools.funcutils import  to_numeric_dtypes , smart_format, key_checker
from ..tools.funcutils import  random_sampling, assert_ratio, split_train_test_by_id
from ..tools.funcutils import  format_to_datetime, is_in_if, validate_feature
from ..tools.box import Boxspace
from ._globals import FORENSIC_BF_DICT, FORENSIC_LABELS_DESCR
from .io import (csv_data_loader, _to_dataframe, DMODULE, 
    description_loader, DESCR, RemoteDataURL ) 

__all__= [ "load_iris",  "load_hlogs", "load_mxs", "load_nlogs", "load_forensic", 
          "load_jrs_bet"]

def load_hlogs (
        *,  return_X_y=False, as_frame =False, key =None,  split_X_y=False, 
        test_ratio =.3 , tag =None, tnames = None , data_names=None, 
         **kws): 
    
    drop_observations =kws.pop("drop_observations", False)
    cf = as_frame 
    key = key or 'h502' 
    # assertion error if key does not exist. 
    available_sets = {
        "h502", 
        "h2601", 
        'h1102',
        'h1104',
        'h1405',
        'h1602',
        'h2003',
        'h2602',
        'h604',
        'h803',
        'h805'
        }
    is_keys = set ( list(available_sets) + ["*"])
    key = key_checker(key, is_keys)
    
    data_file ='h.h5'
    
    #-----------------------------------------------------------
    if not check_file_exists(DMODULE, data_file): 
        # If file does not exist download it from the remote and 
        # save it to the path 
        package_path = str(files(DMODULE).joinpath(data_file))
        URL= os.path.join( RemoteDataURL, data_file) 
        fancier_downloader (URL,data_file, dstpath = os.path.dirname (package_path)
                       )
    #-------------------------------------------------------------- 
    with resources.path (DMODULE , data_file) as p : 
        data_file = str(p)
        
    if key =='*': 
        key = available_sets
        
    if isinstance (key, str): 
        data = pd.read_hdf(data_file, key = key)
    else: 
        data =  pd.concat( [ pd.read_hdf(data_file, key = k) for k in key ]) 

    if drop_observations: 
        data.drop (columns = "remark", inplace = True )
        
    frame = None
    feature_names = list(data.columns [:12] ) 
    target_columns = list(data.columns [12:])
    
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    try : 
        validate_feature(data[target_columns] , tnames)
    except Exception as error: 
        # get valueError message and replace Feature by target 
        msg = (". Acceptable target values are:"
               f"{smart_format(target_columns, 'or')}")
        raise ValueError(str(error).replace(
            'Features'.replace('s', ''), 'Target(s)')+msg)
        
    if  ( 
            split_X_y
            or return_X_y
            ) : 
        as_frame =True 
        
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = tnames, 
            target=data[tnames].values 
            )
        frame = to_numeric_dtypes(frame)
        
    if split_X_y: 
        X, Xt = split_train_test_by_id (data = frame , test_ratio= test_ratio, 
                                        keep_colindex= False )
        y = X[tnames] 
        X.drop(columns =target_columns, inplace =True)
        yt = Xt[tnames]
        Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if cf else (
            X.values, Xt.values, y.values , yt.values )
    
    if return_X_y: 
        data , target = data.values, target.values
        
    if ( 
            return_X_y 
            or cf
            ) : return data, target 
    
    # Loading description
    fdescr = description_loader(descr_module=DESCR, descr_file="hlogs.rst")
    return Boxspace(
        data=data.values,
        target=data[tnames].values,
        frame=data,
        tnames=tnames,
        target_names = target_columns,
        DESCR= fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

load_hlogs.__doc__ = """\
Load hydro-logging dataset for hydrogeophysical analysis.

This dataset contains multi-target data suitable for both classification and 
regression tasks in the context of groundwater studies.

Parameters
----------
return_X_y : bool, default=False
    If True, returns `(data, target)` instead of a Bowlspace object. 
    `data` and `target` are described in more detail below.

as_frame : bool, default=False
    If True, data is a pandas DataFrame with appropriate dtypes (numeric).
    The target is a pandas DataFrame or Series, depending on the number of 
    target columns. If `return_X_y` is True, then both `data` and `target` will 
    be pandas DataFrames or Series.

split_X_y: bool, default=False
    If True, splits the data into training and testing sets based on the 
    `test_size` ratio. Returns the sets as `(X, y)` for training and `(Xt, yt)` 
    for testing.

test_ratio: float, default=0.3
    Ratio for splitting data into training and testing sets. A value of 0.3 
    implies a 30% allocation for testing.

tnames: str, optional
    The name of the target column(s) to retrieve. If None, all target columns 
    are collected for multioutput `y`. For specific classification or regression 
    tasks, it's advisable to specify the relevant target name.

key: str, default='h502'
    Identifier for the specific logging data to fetch. Accepts borehole IDs 
    (e.g., "h2601", "*"). If `key='*'`, aggregates all data into a single frame.

drop_observations: bool, default=False
    If True, drops the `remark` column from the logging data.

Returns
-------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object with attributes:
    - data: {ndarray, DataFrame} 
      Data matrix; pandas DataFrame if `as_frame=True`.
    - target: {ndarray, Series}
      Classification target; pandas Series if `as_frame=True`.
    - feature_names: list
      Names of dataset columns.
    - target_names: list
      Names of target classes.
    - frame: DataFrame
      DataFrame with `data` and `target` if `as_frame=True`.
    - DESCR: str
      Full description of the dataset.
    - filename: str
      Path to data location.

data, target : tuple
    Returned if `return_X_y` is True. Tuple of ndarray: data matrix (n_samples, n_features)
    and target samples (n_samples,).

X, Xt, y, yt : tuple
    Returned if `split_X_y` is True. Tuple of ndarrays for training (X, y) 
    and testing (Xt, yt) sets, split according to `test_ratio`. The shapes are determined as follows:
    \[
    \text{{shape}}(X, y) = \left(1 - \text{{test\_ratio}}\right) \times (n_{\text{{samples}}}, n_{\text{{features}}}) \times 100
    \]
    \[
    \text{{shape}}(Xt, yt) = \text{{test\_ratio}} \times (n_{\text{{samples}}}, n_{\text{{features}}}) \times 100
    \]

Examples
--------
To explore available target columns without specifying any parameters:

>>> from gofast.datasets.dload import load_hlogs
>>> b = load_hlogs()
>>> b.target_names 
['aquifer_group', 'pumping_level', 'aquifer_thickness', ...]

To focus on specific targets 'pumping_level' and 'aquifer_thickness':

>>> _, y = load_hlogs(as_frame=True, tnames=['pumping_level', 'aquifer_thickness'])
>>> list(y.columns)
['pumping_level', 'aquifer_thickness']
"""

def load_nlogs (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None, 
    years=None, 
    split_X_y=False, 
    test_ratio=.3 , 
    tag=None, 
    tnames=None, 
    data_names=None, 
    samples=None, 
    seed =None, 
    shuffle =False, 
    **kws
    ): 

    drop_display_rate = kws.pop("drop_display_rate", True)
    key = key or 'b0' 
    # assertion error if key does not exist. 
    available_sets = {
        "b0", 
        "ns", 
        "hydrogeological", 
        "engineering", 
        "subsidence", 
        "ls"
        }
    is_keys = set ( list(available_sets))
    key = key_checker(key, is_keys, deep_search=True )
    key = "b0" if key in ("b0", "hydrogeological") else (
          "ns" if key in ("ns",  "engineering") else ( 
          "ls" if key in ("ls", "subsidence") else key )
        )
    assert key in (is_keys), (
        f"wrong key {key!r}. Expect {smart_format(is_keys, 'or')}")
    
    # read datafile
    if key in ("b0", "ns"):  
        data_file =f"nlogs{'+' if key=='ns' else ''}.csv" 
    else: data_file = "n.npz"
    
    #-----------------------------------------------------------
    if not check_file_exists(DMODULE, data_file): 
        # If file does not exist download it from the remote and 
        # save it to the path 
        package_path = str(files(DMODULE).joinpath(data_file))
        URL= os.path.join( RemoteDataURL, data_file) 
        fancier_downloader (URL,data_file, dstpath = os.path.dirname (package_path)
                       )
    #-------------------------------------------------------------- 
    with resources.path (DMODULE, data_file) as p : 
        data_file = str(p)
     
    if key=='ls': 
        # use tnames and years 
        # interchangeability 
        years = tnames or years 
        data , feature_names, target_columns= _get_subsidence_data(
            data_file, years = years or "2022",
            drop_display_rate= drop_display_rate )
        # reset tnames to fit the target columns
        tnames=target_columns 
    else: 
        data = pd.read_csv( data_file )
        # since target and columns are alread set 
        # for land subsidence data, then 
        # go for "ns" and "b0" to
        # set up features and target names
        feature_names = (list( data.columns [:21 ])  + [
            'filter_pipe_diameter']) if key=='b0' else list(
                filter(lambda item: item!='ground_height_distance',
                       data.columns)
                ) 
        target_columns = ['static_water_level', 'drawdown', 'water_inflow', 
                          'unit_water_inflow', 'water_inflow_in_m3_d'
                          ] if key=='b0' else  ['ground_height_distance']
    # cast values to numeric 
    data = to_numeric_dtypes( data) 
    samples = samples or "*"
    data = random_sampling(data, samples = samples, random_state= seed, 
                            shuffle= shuffle) 
    # reverify the tnames if given 
    # target columns 
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    try : 
        validate_feature(data[target_columns], tnames)
    except Exception as error: 
        # get valueError message and replace Feature by target
        verb ="s are" if len(target_columns) > 2 else " is"
        msg = (f" Valid target{verb}: {smart_format(target_columns, 'or')}")
        raise ValueError(str(error).replace(
            'Features'.replace('s', ''), 'Target(s)')+msg)
        
    # read dataframe and separate data to target. 
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, tnames = tnames, 
        target=data[tnames].values 
        )
    # for consistency, re-cast values to numeric 
    frame = to_numeric_dtypes(frame)
        
    if split_X_y: 
        
        X, Xt = split_train_test_by_id (
            data = frame , test_ratio= assert_ratio(test_ratio), 
            keep_colindex= False )
        
        y = X[tnames] 
        X.drop(columns =target_columns, inplace =True)
        yt = Xt[tnames]
        Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if as_frame else (
            X.values, Xt.values, y.values , yt.values )

    if return_X_y: 
        return (data.values, target.values) if not as_frame else (
            data, target) 
    
    # return frame if as_frame simply 
    if as_frame: 
        return frame 
    # upload the description file.
    descr_suffix= {"b0": '', "ns":"+", "ls":"++"}
    fdescr = description_loader(
        descr_module=DESCR,descr_file=f"nansha{descr_suffix.get(key)}.rst")

    return Boxspace(
        data=data.values,
        target=target.values,
        frame=frame,
        tnames=tnames,
        target_names = target_columns,
        DESCR= fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )
 
load_nlogs.__doc__ = """\
Load the Nansha Engineering and Hydrogeological Drilling Dataset.

This dataset contains multi-target information suitable for classification or 
regression problems in hydrogeological and geotechnical contexts.

Parameters
----------
return_X_y : bool, default=False
    If True, returns (data, target) as separate objects instead of a 
    Bowlspace object.

as_frame : bool, default=False
    If True, data is returned as a pandas DataFrame with appropriate dtypes. 
    The target is also returned as a DataFrame or Series, based on the number 
    of target columns.

split_X_y: bool, default=False
    If True, data is split into a training set (X, y) and a testing set (Xt, yt) 
    according to the test ratio specified.

test_ratio: float, default=0.3
    Ratio for splitting data into training and testing sets (default is 30%).

tnames: str, optional 
    Name(s) of the target column(s) to retrieve. If None, all target columns 
    are included.

key: str, default='b0'
    Identifier for the type of drilling data to fetch. Options include 
    engineering drilling ('ns') and hydrogeological drilling ('b0').

years: str, default="2022"
    Specific year(s) for land subsidence data, ranging from 2015 to 2022.

samples: int, optional 
    Number of samples to fetch from the dataset. Fetches all data if None.

seed: int, optional
    Seed for the random number generator, used when shuffling data.

shuffle: bool, default=False
    If True, shuffles data before sampling.

drop_display_rate: bool, default=True
    If True, removes the display rate column used for image visualization.

Returns
-------
data : :class:`~gofast.tools.box.Boxspace`
    Dictionary-like object with attributes:
    - data: {ndarray, DataFrame} 
      Data matrix; pandas DataFrame if `as_frame=True`.
    - target: {ndarray, Series}
      Classification target; pandas Series if `as_frame=True`.
    - feature_names: list
      Names of dataset columns.
    - target_names: list
      Names of target classes.
    - frame: DataFrame
      DataFrame with `data` and `target` if `as_frame=True`.
    - DESCR: str
      Full description of the dataset.
    - filename: str
      Path to data location.

data, target : tuple
    Returned if `return_X_y` is True. Tuple of ndarray: data matrix 
    (n_samples, n_features) and target samples (n_samples,).

X, Xt, y, yt : tuple
    Returned if `split_X_y` is True. Tuple of ndarrays for training (X, y) 
    and testing (Xt, yt) sets, split according to `test_ratio`.

Examples
--------
To explore available target columns without specifying any parameters:

>>> from gofast.datasets.dload import load_nlogs
>>> b = load_nlogs()
>>> b.target_names
['static_water_level', 'drawdown', 'water_inflow', ...]

To focus on specific targets 'drawdown' and 'static_water_level':

>>> _, y = load_nlogs(as_frame=True, tnames=['drawdown', 'static_water_level'])
>>> list(y.columns)
['drawdown', 'static_water_level']

To retrieve land subsidence data for specific years with display rate:

>>> n = load_nlogs(key='ls', samples=3, years="2015 2018", drop_display_rate=False)
>>> n.frame.head()
[easting, northing, longitude, 2015, 2018, disp_rate]
"""

def load_bagoue(
        *, return_X_y=False, 
        as_frame=False, 
        split_X_y=False, 
        test_size =.3 , 
        tag=None , 
        data_names=None, 
        **kws
 ):
    cf = as_frame 
    data_file = "bagoue.csv"
    data, target, target_names, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="bagoue.rst", include_headline= True, 
    )
    frame = None
    target_columns = [
        "flow",
    ]
    if split_X_y: 
        as_frame =True 
        
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = target_columns, 
            target=target)
        frame = to_numeric_dtypes(frame)

    if split_X_y: 
        X, Xt = split_train_test_by_id (data = frame , test_ratio= test_size, 
                                        keep_colindex= False )
        y = X.flow ;  X.drop(columns =target_columns, inplace =True)
        yt = Xt.flow ; Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if cf else (
            X.values, Xt.values, y.values , yt.values )
    
    if return_X_y or as_frame:
        return to_numeric_dtypes(data) if as_frame else data , target
    
    frame = to_numeric_dtypes (
        pd.concat ([pd.DataFrame (data, columns =feature_names),
            pd.DataFrame(target, columns= target_names)],axis=1))
    
    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        tnames=target_columns,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

load_bagoue.__doc__="""\
Load the Bagoue dataset. 

The Bagoue dataset is a classic and a multi-class classification
dataset. Refer to the description for more details. 

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a 
    :class:`~gofast.tools.box.Boxspace` object. See below for more information 
    about the `data` and `target` object.
    .. versionadded:: 0.1.2
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
    .. versionadded:: 0.1.1
split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) with the according to the test size ratio.  
test_size: float, default is {{.3}} i.e. 30% (X, y)
    The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
    respectively.
tag, data_names: None
    `tag` and `data_names` do nothing. just for API purpose. They allow 
    to fetch the same data uing the func:`~gofast.datasets.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 

Returns
-------
data: :class:`~gofast.tools.box.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
        .. versionadded:: 0.1.2
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.
    .. versionadded:: 0.1.2
X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of:
        
    .. math:: 
        
        \\text{shape}(X, y) =  1-  \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
        \\text{shape}(Xt, yt)= \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
    where each row representing one sample and each column representing the 
    features. The second ndarray of shape(n_samples,) containing the target 
    samples.
     
Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name::

>>> from gofast.datasets import load_bagoue
>>> d = load_bagoue () 
>>> d.target[[10, 25, 50]]
array([0, 2, 0])
>>> list(d.target_names)
['flow']   
"""

def load_iris(
        *, return_X_y=False, as_frame=False, tag=None, data_names=None, **kws
        ):
    data_file = "iris.csv"
    data, target, target_names, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="iris.rst"
    )
    feature_names = ["sepal length (cm)","sepal width (cm)",
        "petal length (cm)","petal width (cm)",
    ]
    frame = None
    target_columns = [
        "target",
    ]
    #if as_frame:
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, tnames = target_columns, 
        target = target)
        # _to(
        #     "load_iris", data, target, feature_names, target_columns
        # )
    if return_X_y or as_frame:
        return to_numeric_dtypes(data) if as_frame else data , target

    return Boxspace(
        data=data.values,
        target=target,
        frame=frame,
        tnames=target_names,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        )


load_iris.__doc__="""\
Load and return the iris dataset (classification).
The iris dataset is a classic and very easy multi-class classification
dataset.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a BowlSpace object. See
    below for more information about the `data` and `target` object.

as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~gofast.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
Returns
-------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.

    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.

(data, target) : tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.

    
Notes
-----
Fixed two wrong data points according to Fisher's paper.
The new version is the same as in R, but not as in the UCI
Machine Learning Repository.

Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name.
>>> from gofast.datasets import load_iris
>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_names)
['setosa', 'versicolor', 'virginica']
"""    
    
    
def load_mxs (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None,  
    tag =None, 
    samples=None, 
    tnames = None , 
    data_names=None, 
    split_X_y=False, 
    seed = None, 
    shuffle=False,
    test_ratio=.2,  
    **kws): 
    
    drop_observations =kws.pop("drop_observations", False)
    
    target_map= { 
        0: '1',
        1: '11*', 
        2: '2', 
        3: '2*', 
        4: '3', 
        5: '33*'
        }
    
    add = {"data": ('data', ) , '*': (
        'X_train','X_test' , 'y_train','y_test' ), 
        }
    av= {"sparse": ('X_csr', 'ymxs_transf'), 
         "scale": ('Xsc', 'ymxs_transf'), 
         "train": ( 'X_train', 'y_train'), 
         "test": ('X_test', 'y_test'), 
         'numeric': ( 'Xnm', 'ymxs_transf'), 
         'raw': ('X', 'y')
         }
    
    if key is None: 
        key='data'
    data_file ='mxs.joblib'
    with resources.path (DMODULE , data_file) as p : 
        data_file = str(p)
        
    data_dict = joblib.load (data_file )
    # assertion error if key does not exist. 
    available_sets = set (list( av.keys() ) + list( add.keys()))

    msg = (f"key {key!r} does not exist yet, expect"
           f" {smart_format(available_sets, 'or')}")
    assert str(key).lower() in available_sets , msg
    # manage sampling 
    # by default output 50% data 
    samples= samples or .50 
    
    if split_X_y: 
        from ..exlib import train_test_split
        data = tuple ([data_dict [k] for k in add ['*'] ] )
        # concatenate the CSR matrix 
        X_csr = scipy.sparse.csc_matrix (np.concatenate (
            (data[0].toarray(), data[1].toarray()))) 
        y= np.concatenate ((data[-2], data[-1]))
        # resampling 
        data = (random_sampling(d, samples = samples,random_state= seed , 
                                shuffle= shuffle) for d in (X_csr, y ) 
                )
        # split now
        return train_test_split (*tuple ( data ),random_state = seed, 
                                 test_size =assert_ratio (test_ratio),
                                 shuffle = shuffle)
    # Append Xy to Boxspace if 
    # return_X_y is not set explicitly.
    Xy = dict() 
    # if for return X and y if k is not None 
    if key is not None and key !="data": 
        if key not in  av.keys():
            key ='raw'
        X, y =  tuple ( [ data_dict[k]  for k in av [key]] ) 

        X = random_sampling(X, samples = samples,random_state= seed , 
                            shuffle= shuffle)
        y = random_sampling(y, samples = samples, random_state= seed, 
                            shuffle= shuffle
                               )
        if return_X_y: 
            return (  X, y )  if as_frame or key =='sparse' else (
                np.array(X), np.array(y))
        
        # if return_X_y is not True 
        Xy ['X']=X ; Xy ['y']=y 
        # Initialize key to 'data' to 
        # append the remain data 
        key ='data'

    data = data_dict.get(key)  
    if drop_observations: 
        data.drop (columns = "remark", inplace = True )
        
    frame = None
    feature_names = list(data.columns [:13] ) 
    target_columns = list(data.columns [13:])
    
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    data = random_sampling(data, samples = samples, random_state= seed, 
                           shuffle= shuffle)
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = tnames, 
            target=data[tnames].values 
            )
        frame = to_numeric_dtypes(frame)

    return Boxspace(
        data=data.values,
        target=data[tnames].values,
        frame=data,
        tnames=tnames,
        target_names = target_columns,
        target_map = target_map, 
        nga_labels = data_dict.get('nga_labels'), 
        #XXX Add description 
        DESCR= '', # fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        **Xy
        )
    
load_mxs.__doc__="""\
Load the dataset after implementing the mixture learning strategy (MXS).

Dataset is composed of 11 boreholes merged with multiple-target that can be 
used for a classification problem.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a Bowlspace object. See
    below for more information about the `data` and `target` object.
    
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.

split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) based on to the `test_ratio` value.

tnames: str, optional 
    the name of the target to retrieve. If ``None`` the full target columns 
    are collected and compose a multioutput `y`. For a singular classification 
    or regression problem, it is recommended to indicate the name of the target 
    that is needed for the learning task. 
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~gofast.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
    
samples: int,optional 
   Ratio or number of items from axis to fetch in the data. 
   Default = .5 if `samples` is ``None``.

key: str, default='data'
    Kind of MXS data to fetch. Can also be: 
        
        - "sparse": for a compressed sparsed row matrix format of train set X. 
        - "scale": returns a scaled X using the standardization strategy 
        - "num": Exclusive numerical data and exclude the 'strata' feature.
        - "test": test data `X` and `y` 
        - "train": train data `X` and  `y` with preprocessing already performed
        - "raw": for original dataset X and y  with no preprocessing 
        - "data": Default when key is not supplied. It returns 
          the :class:`Bowlspace` objects.
        
    When k is not supplied, "data" is used instead and return a 
    :class:`Bowlspace` objects. where: 
        - target_map: is the mapping of MXS labels in the target y. 
        - nga_labels: is the y predicted for Naive Group of Aquifer. 

drop_observations: bool, default='False'
    Drop the ``remark`` column in the logging data if set to ``True``. 
    
seed: int, array-like, BitGenerator, np.random.RandomState, \
    np.random.Generator, optional
   If int, array-like, or BitGenerator, seed for random number generator. 
   If np.random.RandomState or np.random.Generator, use as given.
   
shuffle: bool, default =False, 
   If ``True``, borehole data should be shuffling before sampling. 
   
test_ratio: float, default is 0.2 i.e. 20% (X, y)
    The ratio to split the data into training (X, y) and testing (Xt, yt) set 
    respectively.
    
Returns
---------
data : :class:`~gofast.tools.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} 
        The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
    target: {ndarray, Series} 
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    target_map: dict, 
       is the mapping of MXS labels in the target y. 
    nga_labels: arryalike 1D, 
       is the y predicted for Naive Group of Aquifer. 
    frame: DataFrame 
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.

X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of
    training and test data whereas `y` and `yt` are training and test labels.
    The number of samples are based on the `test_ratio`. 
 
Examples
--------
>>> from gofast.datasets.dload import load_mxs  
>>> load_mxs (return_X_y= True, key ='sparse', samples ='*')
(<1038x21 sparse matrix of type '<class 'numpy.float64'>'
 	with 8298 stored elements in Compressed Sparse Row format>,
 array([1, 1, 1, ..., 5, 5, 5], dtype=int64))
 
"""  
def _get_subsidence_data (
        data_file, /, 
        years: str="2022", 
        drop_display_rate: bool=... 
        ): 
    """Read, parse features and target for Nanshan land subsidence data
    
    Parameters 
    ------------
    data_file: str, Pathlike object 
       Full path to the object to read.
    years: str, default=2022 
        year of subsidence data collected. To collect the value of subsidence 
        of all years, set ``years="*"``
        
    drop_display_rate: bool, default=False, 
       Rate of display for visualisation in Goldern software. 
       
    Returns 
    --------
    data, feature_names, target_columns: pd.DataFrame, list
      DataFrame and list of features and targets. 
   
    """
    columns =['easting',
             'northing',
             'longitude',
             'latitude',
             '2015',
             '2016',
             '2017',
             '2018',
             '2019',
             '2020',
             '2021',
             '2022',
             'disp_rate'
             ]
    data = read_data ( data_file, columns = columns )
    if drop_display_rate: 
        data.pop('disp_rate')
        columns =columns [: -1]
        # remove display rate if exists while 
        # it is set to True
        if isinstance ( years, str): 
           years = years.replace ("disp", "").replace (
               "_", ' ').replace ("rate", "")
        elif hasattr ( years, '__iter__'): 
            # maybe list etc 
            years = [str(i) for i in years if not (str(i).find(
                "disp") >= 0 or str(i).find("rate")>= 0) ] 

    if years !="*": 
        years = key_checker (years, valid_keys= data.columns ,
                     pattern =r'[#&*@!,;\s-]\s*', deep_search=True)
        years = [years] if isinstance ( years, str) else years 
    else : 
        years = columns[4:] # target columns 
    # recheck duplicates and remove it  
    years = sorted(set ((str(y) for y in years)))
    feature_names = columns[: 4 ]
    target_columns = years 
    data = data [ feature_names + target_columns ]
    
    return  data,  feature_names, target_columns     
    
    
def load_bagoue(
        *, 
        return_X_y=False, 
        as_frame=False, 
        split_X_y=False, 
        test_size =.3 , 
        tag=None , 
        data_names=None,
        **kws
 ):
    cf = as_frame 
    data_file = "bagoue.csv"
    data, target, target_names, feature_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="bagoue.rst", include_headline= True, 
    )
    frame = None
    target_columns = [
        "flow",
    ]
    if split_X_y: 
        as_frame =True 
        
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = target_columns, 
            target=target)
        frame = to_numeric_dtypes(frame)

    if split_X_y: 
        X, Xt = split_train_test_by_id (data = frame , test_ratio= test_size, 
                                        keep_colindex= False )
        y = X.flow ;  X.drop(columns =target_columns, inplace =True)
        yt = Xt.flow ; Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if cf else (
            X.values, Xt.values, y.values , yt.values )
    

    if as_frame and not return_X_y: 
        return frame 

    if return_X_y:
        return data, target
    
    frame = to_numeric_dtypes (
        pd.concat ([pd.DataFrame (data, columns =feature_names),
            pd.DataFrame(target, columns= target_names)],axis=1))
    
    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        tnames=target_columns,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

load_bagoue.__doc__ = """\
Load the Bagoue dataset.

The Bagoue dataset is a classic, multi-class classification dataset commonly
used in water-related studies. For detailed information, refer to the dataset
description.

Parameters
----------
return_X_y : bool, default=False
    If True, returns (data, target) instead of a Boxspace object. 
    Refer to below for detailed structure of data and target.
as_frame : bool, default=False
    If True, returns data as a pandas DataFrame with appropriate dtypes.
    The target is returned as a DataFrame or Series based on target column count.
split_X_y: bool, default=False
    If True, splits data into a training set (X, y) and a testing set (Xt, yt) 
    according to the test size ratio.
test_size: float, default=0.3
    Ratio for splitting data into training and testing sets.
tag, data_names: None
    Parameters for API consistency. Do not modify dataset fetching.

Returns
-------
data : Boxspace
    Dictionary-like object with the following attributes:
    - data : {ndarray, DataFrame}
      Data matrix. DataFrame if `as_frame=True`.
    - target : {ndarray, Series}
      Classification target. Series if `as_frame=True`.
    - feature_names : list
      Names of dataset columns.
    - target_names : list
      Names of target classes.
    - frame : DataFrame
      Complete DataFrame with data and target, present if `as_frame=True`.
    - DESCR : str
      Full dataset description.
    - filename : str
      Path to dataset location.

data, target : tuple
    Returned if `return_X_y` is True. Tuple contains data matrix 
    (n_samples, n_features) and target samples (n_samples,).

X, Xt, y, yt : tuple
    Returned if `split_X_y` is True. Tuple contains training set
    (X, y) and testing set (Xt, yt),
    split according to `test_size`.

Examples
--------
To explore a subset of samples:

>>> from gofast.datasets import load_bagoue
>>> d = load_bagoue()
>>> d.target[[10, 25
"""   

def load_forensic( *, 
       return_X_y=False, 
       as_frame=False, 
       key=None, 
       split_X_y=False, 
       test_size =.3 ,  
       tag=None , 
       data_names=None, 
       **kws
):
   cf = as_frame 
   key = key or 'preprocessed'
   key = key_checker(key, valid_keys=("raw", "preprocessed"), 
                     deep_search=True )
   data_file = f"forensic_bf{'+'if key=='preprocessed' else ''}.csv"

   with resources.path (DMODULE, data_file) as p : 
       file = str(p)
   data = pd.read_csv ( file )
   
   frame = None
   target_columns = [
       "dna_use_terrorism_fight",
   ]
   feature_names= is_in_if(data, items=target_columns, return_diff=True)
   
   frame, data, target = _to_dataframe(
        data, feature_names = feature_names, tnames = target_columns, 
        )
   frame = format_to_datetime(to_numeric_dtypes(frame), date_col ='timestamp')

   if split_X_y: 
       X, Xt = split_train_test_by_id (data = frame , test_ratio= test_size, 
                                       keep_colindex= False )
       y = X.flow ;  X.drop(columns =target_columns, inplace =True)
       yt = Xt.flow ; Xt.drop(columns =target_columns, inplace =True)
       
       return  (X, Xt, y, yt ) if cf else (
           X.values, Xt.values, y.values , yt.values )
   
   if as_frame and not return_X_y: 
       return frame 

   if return_X_y:
       return data, target
   
   fdescr = description_loader(
       descr_module=DESCR,descr_file=data_file.replace (".csv", ".rst"))
   
   return Boxspace(
       data=data,
       target=target,
       frame=frame,
       tnames=target_columns,
       target_names=target_columns,
       DESCR=fdescr,
       feature_names=feature_names,
       filename=data_file,
       data_module=DMODULE,
       labels_descr=FORENSIC_LABELS_DESCR,
       colums_descr= FORENSIC_BF_DICT
   )
load_forensic.__doc__="""\
Load and return the forensic dataset for criminal investigation studies.

This function provides access to a forensic dataset, which includes public 
opinion and knowledge regarding DNA databases, their potential use in criminal
investigations, and concerns related to privacy and misuse. The dataset is 
derived from a study on the need for a forensic DNA database in the Sahel region. 
It comes in two forms: raw and preprocessed. The raw data includes the original 
responses, while the preprocessed data contains encoded and cleaned information.

Parameters
----------
return_X_y : bool, default=False
    If True, returns `(data, target)` as separate objects. Otherwise, returns 
    a Bunch object.
as_frame : bool, default=False
    If True, the data and target are returned as a pandas DataFrame and Series, 
    respectively. This is useful for further analysis and visualization in a 
    tabular format.
key : {'raw', 'preprocessed'}, default='preprocessed'
    Specifies which version of the dataset to load. 'raw' for the original 
    dataset and 'preprocessed' for the processed dataset with encoded 
    categorical variables.
split_X_y : bool, default=False
    If True, splits the dataset into training and testing sets based on the 
    `test_size` parameter. This is useful for model training and evaluation 
    purposes.
test_size : float, default=0.3
    The proportion of the dataset to include in the test split. It should be 
    between 0.0 and 1.0 and represent the size of the test dataset relative to 
    the original dataset.
tag : str or None, optional
    An optional tag to filter or categorize the data. Useful for specific 
    analyses within the dataset.
data_names : list of str or None, optional
    Specific names of datasets to be loaded. If None, all datasets are loaded.
**kws : dict, optional
    Additional keyword arguments to pass to the data loading function. Allows 
    customization of the data loading process.

Returns
-------
boxspace : Bunch object or tuple
    The function returns a Bunch object when `return_X_y` is False. When True,
    returns `(data, target)` 
    as separate objects. The Bunch object has the following attributes:
    - data : ndarray, shape (n_samples, n_features)
      The data matrix with features.
    - target : ndarray, shape (n_samples,)
      The classification targets.
    - frame : DataFrame
      A DataFrame with `data` and `target` when `as_frame=True`.
    - DESCR : str
      The full description of the dataset.
    - feature_names : list
      Names of the feature columns.
    - target_names : list
      Names of the target columns.
    - target_columns : list
      Column names in the dataset used as targets.
    - labels_descr, columns_descr : dict
      Formatted dictionaries providing descriptions for categorical labels.

Examples
--------
>>> from gofast.datasets import load_forensic
>>> forensic_data = load_forensic(key='raw', as_frame=True)
>>> print(forensic_data.frame.head())

References
----------
Detailed dataset descriptions can be found in the forensic_bf.rst and 
forensic_bf+.rst files. These documents provide insights into the dataset's 
creation, structure, and attributes.
"""

def load_jrs_bet(
    *, 
    return_X_y=False, 
    as_frame=False,
    key=None, 
    split_X_y=False, 
    test_size=0.2, 
    tag=None, 
    data_names=None, 
    N=5, 
    seed=None, 
    **kws
 ):
    # Validating the key
    key = key or 'classic'
    key = key_checker(key, valid_keys=("raw", "classic", "neural"),
                      deep_search=True )

    # Loading the dataset
    data_file = "jrs_bet.csv"
    with resources.path(DMODULE, data_file) as p: 
        file = str(p)
    data = pd.read_csv(file)

    # Preparing the dataset
    results = _prepare_jrs_bet_data(
        data, key=key, N=N, split_X_y=split_X_y, return_X_y=return_X_y,
        test_size=test_size, seed=seed
        )

    if split_X_y or return_X_y: 
        return results

    # Processing the data for return
    target_columns=[] 
    if key != 'neural': 
       target_columns = ["target" if key == 'classic' else 'winning_numbers']
       feature_names = is_in_if(results.columns, items=target_columns,
                                return_diff=True)
    else:feature_names= list(results.columns )
    frame, data, target = _to_dataframe(results, feature_names=feature_names,
                                        tnames=target_columns)

    if key in ("neural", "raw"):
        frame = format_to_datetime(to_numeric_dtypes(frame), date_col='date')

    if as_frame:
        return frame

    # Loading description
    fdescr = description_loader(descr_module=DESCR, descr_file=data_file.replace(
        ".csv", f"{'+' if key in ('classic', 'neural') else ''}.rst"))

    # Returning a Boxspace object
    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        tnames=target_columns,
        target_names=target_columns,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE
    )

load_jrs_bet.__doc__="""\
Load and prepare the jrs_bet dataset for machine learning applications.

This function allows for the loading and processing of the `jrs_bet` dataset, 
a rich collection of betting data. The dataset, in its raw form, includes 
information such as dates, locations, and winning numbers. Depending on the 
specified `key`, it can be prepared for classical machine learning models 
or neural network models, with each approach entailing specific preprocessing 
steps to optimize the dataset for the chosen model type.

Parameters
----------
return_X_y : bool, optional
    If True, the function returns the features (X) and target (y) as separate
    objects, rather than a combined dataset. Useful for model training and 
    evaluation. By default, this is set to False.
as_frame : bool, optional
    If True, the data is returned as a pandas DataFrame. This is beneficial for 
    exploratory data analysis and when working with data manipulation libraries. 
    By default, this is set to False.
key : str, optional
    Specifies the type of data preparation to apply. Options include 'raw' for 
    no preprocessing, 'classic' for preparation suitable for classical machine 
    learning models, and 'neural' for preparation tailored to neural network models.
    Defaults to None, which treats the data as 'classic'.
split_X_y : bool, optional
    If True, the dataset is split into features (X) and target (y) during the 
    preparation process. This is especially useful when the dataset needs to be 
    directly fed into a machine learning algorithm. By default, this is set to False.
test_size : float, optional
    Defines the proportion of the dataset to be used as the test set. This is 
    important for evaluating the performance of machine learning models. 
    By default, it is set to 0.2.
tag : str, optional
    A custom tag that can be used for dataset processing. This allows for 
    additional customization or identification during the data preparation process.
    By default, this is set to None.
data_names : list, optional
    Custom names for the data columns. This allows for personalization of the 
    dataset column names for clarity or specific requirements. By default, 
    this is set to None.
N : int, optional
    Specifies the number of past draws to consider for LSTM models, useful 
    in the context of the 'neural' key. This parameter is crucial for time-series 
    prediction tasks. By default, it is set to 5.
seed : int, optional
    The seed for the random number generator, which is important for reproducibility 
    in tasks like dataset splitting. By default, this is set to None.
kws : dict
    Additional keyword arguments for more advanced or specific dataset 
    preparation needs.

Returns
-------
boxspace : Bunch object or tuple
    Depending on the `return_X_y` parameter, the function returns either a 
    Bunch object (when False) or a tuple of `(data, target)` (when True). 
    The Bunch object contains several attributes providing insights into the 
    dataset, including the data matrix (`data`), classification targets (`target`), 
    a combined DataFrame (`frame` if `as_frame` is True), a full description 
    (`DESCR`), feature names (`feature_names`), and target names (`target_names`).

Examples
--------
>>> from gofast.datasets import load_jrs_bet 
>>> data_box = load_jrs_bet(key="raw")
>>> print(data_box.frame.head())

>>> df = load_jrs_bet(as_frame=True, key='neural')
>>> print(df.head())

>>> X, y = load_jrs_bet(return_X_y=True)
>>> print(X.head(), y.head())

Notes
-----
The jrs_bet dataset is a versatile collection of data suitable for various 
machine learning applications. The raw dataset serves as a foundational bedrock 
for betting analysis. When preprocessed for classical machine learning, it 
undergoes transformation to facilitate models like logistic regression, decision 
trees, or random forests. For neural network applications, especially RNNs like 
LSTM, the dataset is structured to capture temporal patterns and dependencies, 
crucial for deep learning techniques.
"""
def _prepare_jrs_bet_data(data, /, key='classic', N=5, split_X_y=False,
                         return_X_y=False, test_size=0.2, seed=None
                         ):
    """
    Prepare the jrs_bet dataset for machine learning models.

    This function preprocesses the jrs_bet dataset for use in classical machine
    learning or deep neural network models, specifically LSTM.

    Parameters
    ----------
    data : DataFrame
        The jrs_bet dataset.
    key : str, optional
        The type of model to prepare the data for ('classic' or 'neural'),
        by default 'classic'.
    N : int, optional
        The number of past draws to consider for LSTM, by default 5.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y),
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    seed : int, optional
        The seed for the random number generator, by default None.

    Returns
    -------
    DataFrame or tuple of DataFrames
        Processed dataset suitable for the specified model type.
    """
    # Preprocessing steps
    data.replace('HOLIDAY', pd.NA, inplace=True)
    data.dropna(inplace=True)
    data['date'] = pd.to_datetime(data['date'])

    # Split and convert winning numbers
    winning_numbers = data['winning_numbers'].str.split('-', expand=True)
    winning_numbers = winning_numbers.apply(pd.to_numeric)

    # Prepare data based on the specified model type
    if key in 'classical':
        return _prepare_for_classical_ml(
            winning_numbers, split_X_y=split_X_y, return_X_y=return_X_y, 
            seed=seed, test_size=test_size
        )
    if key =="neural":
        return _prepare_for_neural_networks(
            winning_numbers, data=data, N=N, split_X_y=split_X_y, 
            return_X_y=return_X_y, test_size=test_size
        )
    
    return data 

def _prepare_for_neural_networks(
        winning_numbers, /, data, N=5, split_X_y=False, 
        return_X_y=False, test_size=0.2
   ):
    """
    Prepare the jrs_bet dataset specifically for recurrent neural network models.

    This function processes the winning_numbers from the jrs_bet dataset 
    for application in recurrent neural network models such as LSTM.

    Parameters
    ----------
    winning_numbers : DataFrame
        Winning numbers from the jrs_bet dataset.
    data : DataFrame
        The original jrs_bet dataset with additional columns such as 'date'.
    N : int, optional
        The number of past draws to use for sequence generation, by default 5.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y), 
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    test_size : float, optional
        The proportion of the dataset to include in the test split,
        by default 0.2.

    Returns
    -------
    DataFrame or tuple of arrays
        Prepared sequences and corresponding targets suitable for RNN models.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Concatenate date and winning numbers
    historical_data = pd.concat([data['date'], winning_numbers], axis=1)
    
    if not split_X_y and not return_X_y: 
        return historical_data

    # Generating sequences for LSTM
    sequences = [winning_numbers.iloc[i:i+N].values.flatten()
                 for i in range(len(winning_numbers) - N)]

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    sequences_normalized = scaler.fit_transform(sequences)

    # Preparing features and targets
    X = np.array(sequences_normalized)
    y = np.array([sequences_normalized[i+1] for i in range(
        len(sequences_normalized)-1)])

    # Reshaping for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Data splitting
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (X, y) if return_X_y else (X_train, X_test, y_train, y_test)
 
def _prepare_for_classical_ml(
        winning_numbers, /, split_X_y=False, return_X_y=False,
        seed=None, test_size=0.2):
    """
    Prepare the jrs_bet dataset for classical machine learning models.

    This function processes the winning_numbers from the jrs_bet dataset for 
    application in classical machine learning models.

    Parameters
    ----------
    winning_numbers : DataFrame
        Winning numbers from the jrs_bet dataset.
    split_X_y : bool, optional
        If True, split the dataset into features (X) and target (y), 
        by default False.
    return_X_y : bool, optional
        If True, return features (X) and target (y) instead of the whole 
        dataset, by default False.
    seed : int, optional
        The seed for the random number generator, by default None.
    test_size : float, optional
        The proportion of the dataset to include in the test split,
        by default 0.2.

    Returns
    -------
    DataFrame or tuple of DataFrames
        Prepared dataset suitable for classical machine learning models.
    """
    from sklearn.model_selection import train_test_split 
    # Frequency and recency analysis
    frequency_analysis = winning_numbers.apply(pd.Series.value_counts
                                               ).fillna(0).sum(axis=1)
    recency = {number: index for index, row in winning_numbers.iterrows() 
               for number in row}

    # Converting recency to DataFrame and merging with frequency data
    recency_df = pd.DataFrame.from_dict(
        recency, orient='index', columns=['most_recent_draw']
        ).reset_index().rename(columns={'index': 'number'})
    features_df = pd.merge(frequency_analysis.reset_index().rename(
        columns={'index': 'number', 0: 'frequency'}), recency_df, on='number')

    # Target data preparation
    all_numbers = set(range(1, 100))
    target_data = winning_numbers.isin(all_numbers).any(
        axis=1).astype(int).to_frame('target')

    # Merging features with target data
    ml_dataset = pd.merge(features_df, target_data,
                          left_on='number', right_index=True)
    ml_dataset['target'] = ml_dataset['target'].shift(-1)

    # Final dataset preparation
    final_ml_dataset = ml_dataset.dropna().drop(columns=['number'])

    if not split_X_y and not return_X_y: 
        return final_ml_dataset

    # Dataset splitting
    X = final_ml_dataset.drop('target', axis=1)
    y = final_ml_dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    
    return (X, y) if return_X_y else (X_train, X_test, y_train, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
