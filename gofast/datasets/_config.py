# -*- coding: utf-8 -*-
#   Author: LKouadio <etanoyau@gmail.com>
#   License: BSD-3-Clause

"""
Config Dataset Module
=====================

The `Config Dataset` module serves as a dynamic and flexible gateway for accessing 
and processing a variety of datasets, tailored to specific requirements in data 
analysis and machine learning contexts. Utilizing the `_fetch_data` function, this 
module enables users to fetch data corresponding to different stages of data processing 
by specifying a 'tag'. Each tag represents a unique level or type of data processing 
and provides access to a specific form of the dataset, whether it be raw, preprocessed, 
encoded, or prepared for advanced analyses such as PCA or dimensionality reduction.

This module is particularly adept at adapting to various dataset structures, as 
demonstrated with the Bagoue dataset. Users can effortlessly retrieve original data, 
stratified samples, cleaned datasets, or specific portions like the test set, 
simply by specifying the relevant tag. The module ensures compatibility of datasets 
with different stages of data analysis pipelines, from initial exploratory analysis 
to advanced modeling. This functionality streamlines the data preparation workflow 
and enhances the accessibility and usability of diverse datasets, making it an 
invaluable tool in data-driven research and applications.
"""

import re
import joblib
from importlib import resources
import pandas as pd
from sklearn.model_selection import train_test_split
from .io import DMODULE
from ..tools.funcutils import smart_format
from ..exceptions import DatasetError
from .._gofastlog import gofastlog

_logger = gofastlog().get_gofast_logger(__name__)

__all__ = ['_fetch_data']

_BTAGS = (
    'preprocessed',
    'fitted',
    'analysed',
    'encoded',
    'codified',
    'pipe',
    'prepared'
)
# Regex for parsing tags
TAG_REGEX = re.compile(r'|'.join(_BTAGS) + '|origin', re.IGNORECASE)

# Error messages
_ERROR_MSGS = {
    "origin": "Can't fetch an original data <- dict contest details.",
    "pipe": "Can't build default transformer pipeline: <- 'default pipeline'",
    **{key: f"Can't fetch {key} data: <- 'X' & 'y'" for key in _BTAGS if key != 'pipe'}
}

def _fetch_data(tag, data_names=[], **kwargs):
    """
    Fetch dataset based on a specified tag and additional parameters.

    Parameters
    ----------
    tag : str
        The tag representing the dataset processing stage.
    data_names : list, optional
        List of available data names.
    kwargs : dict
        Additional keyword arguments for data loading.

    Returns
    -------
    Object
        The loaded dataset object or a DataFrame.

    Raises
    ------
    DatasetError
        If the tag name is unknown or data loading fails.
    """
    tag = str(tag).lower()
    is_test = 'test' in tag
    processing_mode = _determine_processing_mode(tag, data_names)
    try:
        with resources.path(DMODULE, 'b.pkl') as p:
            _BAG = joblib.load(str(p))

        data = _BAG.get(processing_mode)
        if processing_mode =='pipe':
            return data
 
        X, y = data
        if is_test:
            return _handle_test_data_split(X, y, **kwargs)
    
        return _prepare_final_output(X, y, kwargs.get('return_X_y', False))
    except Exception as e:
        _logger.error(_ERROR_MSGS.get(processing_mode, str(e)))
        raise DatasetError(f"Error in processing data for tag '{tag}': {str(e)}")

def _determine_processing_mode(tag, data_names):
    """
    Determine the data processing mode based on the tag.

    Parameters
    ----------
    tag : str
        The tag representing the dataset processing stage.

    Returns
    -------
    str
        The determined processing mode.
    """
    if any(tag.find(t) >= 0 for t in ('analys', 'scal')):
        return 'analysed'
    if any(tag.find(t) >= 0 for t in ('mid', 'semi', 'preprocess', 'fit')):
        return 'preprocessed'
    if any(tag.find(t) >= 0 for t in ('codif', 'categorized', 'prepared')):
        return 'codified'
    if any(tag.find(t) >= 0 for t in ('sparse', 'csr', 'encoded')):
        return 'encoded'

    match = TAG_REGEX.search(tag)
    if match:
        return match.group()

    raise DatasetError(f"Unknown tag-name '{tag}'. Available tags "
                       f"are: {smart_format(data_names, 'or')}")

def _handle_test_data_split(X, y, **kwargs):
    """
    Handle the data split for test data.

    Parameters
    ----------
    X : DataFrame
        Feature data.
    y : Series
        Target data.
    kwargs : dict
        Additional keyword arguments for train_test_split.

    Returns
    -------
    tuple
        Split test data (X_test, y_test) or (X_train, X_test, y_train, y_test).
    """
    test_size = kwargs.get('test_size', 0.3)
    random_state = kwargs.get('random_state', None)
    split_X_y = kwargs.get('split_X_y', False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return (X_train, X_test, y_train, y_test) if split_X_y else (X_test, y_test)

def _prepare_final_output(X, y, return_X_y):
    """
    Prepare the final output format of the data.

    Parameters
    ----------
    X : DataFrame
        Feature data.
    y : Series
        Target data.
    return_X_y : bool
        Flag to determine the return format.

    Returns
    -------
    DataFrame or tuple
        Combined DataFrame or separate (X, y) depending on return_X_y.
    """
    if return_X_y:
        return X, y
    else:
        return pd.concat((X, y), axis=1)
 
_fetch_data.__doc__ += """\

More Information
----------------
Fetch dataset based on a specified 'tag', which corresponds to each level 
of data processing.

Examples of Retrieving the Bagoue Dataset:
For the Bagoue dataset, the `tag` refers to the stage of data processing. 
There are various options to retrieve data, such as:

- 'original': Retrieves original or raw data. Returns a dictionary of 
   detailed context. 
  Example usage to get a DataFrame:: 
      
      >>> fetch_data('bagoue original').get('data=df')

- 'stratified': Retrieves stratification data.

- 'mid', 'semi', 'preprocess', 'fit': Retrieves data cleaned with attribute 
   experience combinations.

- 'pipe': Retrieves the default pipeline created during data preparation.

- 'analyses', 'pca', 'reduce dimension': Retrieves data with text attributes 
   encoded using the ordinal encoder, plus attribute combinations.

- 'test': Retrieves stratified test set data.

Depending on the 'tag' used, the data returned can be one of the following:

- 'data': The original dataset.

- 'X', 'y': The stratified training set and its corresponding target labels.

- 'X0', 'y0': Data cleaned by dropping useless features and applying numerical 
   attribute combinations, if applicable.

- 'X_prepared', 'y_prepared': Data prepared after applying all transformations 
   via the transformer (pipeline).

- 'XT', 'yT': The stratified test set and its corresponding test labels.

- '_X': The stratified training set used for data analysis. This set contains 
  non-sparse matrices. Categorical text attributes are converted using an 
  Ordinal Encoder.

- '_pipeline': The default pipeline used for data processing.

"""

# pickle bag data details:
# Python.__version__: 3.10.6 , 
# scikit_learn_vesion_: 1.1.3 
# pandas_version : 1.4.4. 
# numpy version: 1.23.3
# scipy version:1.9.3 
    
    
    
    
    
    
    
    
     
