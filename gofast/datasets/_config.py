# -*- coding: utf-8 -*-
#   create date: Thu Sep 23 16:19:52 2021
#   Author: L.Kouadio 
#   License: BSD-3-Clause

"""
Dataset 
==========
Fetch data set from the local machine. If data does not exist, retrieve it 
from the remote (repository or zenodo record ) 
"""
import re
import joblib
from importlib import resources 
import pandas as pd 
from .io import DMODULE 
from ..tools.funcutils import ( 
    smart_format 
    )
from ..exceptions import DatasetError
from .._gofastlog import gofastlog

_logger = gofastlog().get_gofast_logger(__name__)


__all__=['_fetch_data']

_BTAGS = ( 
    'preprocessed', 
    'fitted',
    'analysed', 
    'encoded', 
    'codifided', 
    'pipe',
    'prepared'
    )

_msg =dict (
    origin = ("Can't fetch an original data <- dict contest details." ), 
    )

regex = re.compile ( r'|'.join(_BTAGS ) + '|origin' , re.IGNORECASE
                    ) 
for key in _BTAGS : 
    _msg[key] = (
        "Can't build default transformer pipeline: <-'default pipeline' "
        )  if key =='pipe' else (
            f"Can't fetching {key} data: <-'X' & 'y' "
            )
          
try : 
    with resources.path (DMODULE, 'b.pkl') as p : 
        data_file = str(p) # for consistency
        _BAG = joblib.load (data_file)

except :
    pass 

def _fetch_data(tag, data_names=[] , **kws): 
    # PKGS ="gofast.etc"
    Xy=None 
    tag = str(tag)
    is_test_in = True if tag.lower().find('test')>=0 else False 
    
    if _tag_checker(tag.lower()): 
        pm = 'analysed'
    elif _tag_checker(tag.lower(), ('mid','semi', 'preprocess', 'fit')):
        pm='preprocessed'
    elif _tag_checker(tag.lower(), ('codif','categorized', 'prepared')): 
        pm ='codified'
    elif _tag_checker(tag.lower(), ('sparse','csr', 'encoded')):
        pm='encoded'
    else : 
        pm =regex.search (tag)
        if pm is None: 
            data_names+= list(_BTAGS)
            msg = (f"Unknow tag-name {tag!r}. None dataset is stored"
                f" under the name {tag!r}. Available tags are: "
                f"{smart_format (data_names, 'or')}"
                )
            raise DatasetError(msg)
        pm= pm.group() 
    
    try: 
        with resources.path (DMODULE, 'b.pkl') as p : 
            data_file = str(p) # for consistency
            _BAG = joblib.load (data_file)

        Xy= _BAG.get(pm)
        
        return_X_y= kws.pop('return_X_y', False)
        kind=kws.pop('kind', None ) 
        
        if str(kind).lower().strip().find ("bin")>=0: 
            # create a binary target 
            X, y = Xy 
            y = y.apply ( lambda v: 1 if v !=0 else v, convert_dtype =True )
            # rebuild the tuple 
            Xy =(X, y)
  
        if pm in ('encoded', 'pipe'): 
            return Xy 
        
        if is_test_in: 
            random_state = kws.pop('random_state', None )
            test_size = kws.pop('test_size', .3 )
            split_X_y = kws.pop("split_X_y", False ) 
            # tag=None , data_names=None, **kws
            from sklearn.model_selection import train_test_split 
            
            X, y = Xy 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y , test_size = test_size ,random_state =random_state )
            
            if split_X_y: 
                return X_train, X_test, y_train, y_test
            
            return X_test, y_test 
            
    except : 
            _logger.error (_msg[pm])
            
    if not return_X_y:
        X, y = Xy 
        Xy = pd.concat ((X, y), axis =1 ) 

    return Xy

def _tag_checker (param, tag_id= ('analys', "scal") 
                      # out = 'analysed'
                      ):
    for ix in tag_id: 
        if param.lower().find(ix)>=0:
            return True
    return False 
   
    
_fetch_data.__doc__ ="""\
Fetch dataset from 'tag'. A tag correspond to each level of data 
processing. 

Experiment an example of retrieving Bagoue dataset.

Parameters 
------------
tag: str,  
    stage of data processing. Tthere are different options to retrieve data
    Could be:
        
    * ['original'] => original or raw data -& returns a dict of details 
        contex combine with get method to get the dataframe like::
            
            >>> fetch_data ('bagoue original').get ('data=df')
            
    * ['stratified'] => stratification data
    * ['mid' |'semi'|'preprocess'|'fit']=> data cleaned with 
        attributes experience combinaisons.
    * ['pipe']=>  default pipeline created during the data preparing.
    * ['analyses'|'pca'|'reduce dimension']=> data with text attributes
        only encoded using the ordinal encoder +  attributes  combinaisons. 
    * ['test'] => stratified test set data

       
Returns
-------
    `data`: Original data 
    `X`, `y` : Stratified train set and training target 
    `X0`, `y0`: data cleaned after dropping useless features and combined 
        numerical attributes combinaisons if ``True``
    `X_prepared`, `y_prepared`: Data prepared after applying  all the 
       transformation via the transformer (pipeline). 
    `XT`, `yT` : stratified test set and test label  
    `_X`: Stratified training set for data analysis. So None sparse
        matrix is contained. The text attributes (categorical) are converted 
        using Ordianal Encoder.  
    `_pipeline`: the default pipeline.    
    
"""

# pickle bag data details:
# Python.__version__: 3.10.6 , 
# scikit_learn_vesion_: 1.1.3 
# pandas_version : 1.4.4. 
# numpy version: 1.23.3
# scipy version:1.9.3 
 
    
    
    
    
    
    
    
    
    
    
     
