# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
from __future__ import annotations, print_function 
import os
import re
import h5py
import copy
import time
import shutil 
import pathlib
import warnings 
import threading
import subprocess
from scipy import stats
from six.moves import urllib 
from joblib import Parallel, delayed
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm

from .._typing import Any,  List, NDArray, DataFrame, Optional, Series 
from .._typing import Dict, Union, TypeGuard, Tuple, ArrayLike
from ..decorators import deprecated
from ..exceptions import FileHandlingError 
from ..property import  Config
from .funcutils import is_iterable, ellipsis2false,smart_format, validate_url 
from .funcutils import to_numeric_dtypes, assert_ratio, exist_features
from .funcutils import normalize_string
from ._dependency import import_optional_dependency 
from .validator import array_to_frame, build_data_if, is_frame 
from .validator import check_consistent_length

def summarize_text_columns(
    data: DataFrame, /, 
    column_names: List[str], 
    stop_words: str = 'english', 
    encode: bool = False, 
    drop_original: bool = False, 
    compression_method: Optional[str] = None,
    force: bool=False, 
    ) -> TypeGuard[DataFrame]:
    """
    Applies extractive summarization to specified text columns in a pandas
    DataFrame. 
    
    Each text entry in the specified columns is summarized to its most 
    representative sentence based on TF-IDF scores, considering the provided 
    stop words. The DataFrame is then updated with these summaries. If the 
    text entry is too short for summarization, it remains unchanged. Optionally,
    encodes and compresses the summaries.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the text data to be summarized.
    column_names : List[str]
        The list of column names in the DataFrame that contains the text to be 
        summarized.
    stop_words : str or list, optional
        Either a string denoting the language for which to use the pre-built 
        stop words list ('english', 'french', etc.), or a list of stop words 
        to use for filtering during the TF-IDF vectorization (default is 'english').
    encode : bool, default False
        If True, adds a new column for each text column containing the TF-IDF 
        encoding of the summary.
    drop_original : bool, default False
        If True, drops the original text columns after summarization and 
        encoding.
    compression_method : str, optional
       Method to compress the encoded vector into a single numeric value.
       Options include 'sum', 'mean', 'norm'. If None, the full vector is returned.
    force: bool, default=False 
       Construct a temporary dataFrame if the numpy array is passed instead.
    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified columns' text summarized and 
        optionally encoded. Original text columns are dropped if drop_original 
        is True.

    Raises
    ------
    ValueError
        If any of the specified column_names do not exist in the DataFrame.
    TypeError
        If the input df is not a pandas DataFrame or if any of the specified 
        columns are not of type str.

    Examples
    --------
    >>> from gofast.tools.baseutils import summarize_text_columns
    >>> data = {
    ...    'id': [1, 2],
    ...    'column1': [
    ...        "Sentence one. Sentence two. Sentence three.",
    ...        "Another sentence one. Another sentence two. Another sentence three."
    ...    ],
    ...    'column2': [
    ...        "More text here. Even more text here.",
    ...        "Second example here. Another example here."
    ...    ]
    ... }
    >>> df = pd.DataFrame(data)
    >>> summarized_df = summarize_text_columns(df, ['column1', 'column2'], 
    ...                stop_words='english', encode=True, drop_original=True, 
    ...                compression_method='mean')
    >>> print(summarized_df.columns)
        id  column1_encoded  column2_encoded
     0   1              1.0         1.000000
     1   2              1.0         0.697271
     Column 'column1' does not exist in the DataFrame.
    >>> # Make sure the column name is exactly "column1"
    >>> if "column1" in df.columns:
    ...     summarized_df = summarize_text_columns(
    ...      df, ['column1'], stop_words='english', encode=True, drop_original=True)
    ... else:
    ...    print("Column 'column1' does not exist in the DataFrame.")
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _summarize_and_encode(text):
        sentences = text.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence]
        if len(sentences) <= 1:
            return text, None
    
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # The similarity matrix is square, so we need to avoid comparing a 
        # sentence to itself
        np.fill_diagonal(similarity_matrix, 0)
        # Sum the similarities of each sentence to all others
        sentence_scores = similarity_matrix.sum(axis=1)
        # Identify the index of the most important sentence
        most_important_sentence_index = np.argmax(sentence_scores)
        summary = sentences[most_important_sentence_index]
        encoding = tfidf_matrix[most_important_sentence_index].todense() if encode else None
        if encoding is not None and compression_method:
            if compression_method == 'sum':
                encoding = np.sum(encoding)
            elif compression_method == 'mean':
                encoding = np.mean(encoding)
            elif compression_method == 'norm':
                encoding = np.linalg.norm(encoding)
            else: 
                raise ValueError(
                    f"Unsupported compression method: {compression_method}")
        return summary, encoding
        
    def _summarize_and_encode0(text):
        sentences = text.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence]
        if len(sentences) <= 1:
            return text, None
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).flatten()
        most_important_sentence_index = np.argmax(similarity_matrix[:-1])
        summary = sentences[most_important_sentence_index]
        encoding = tfidf_matrix[most_important_sentence_index].todense() if encode else None

        if encoding is not None and compression_method:
            if compression_method == 'sum':
                encoding = np.sum(encoding)
            elif compression_method == 'mean':
                encoding = np.mean(encoding)
            elif compression_method == 'norm':
                encoding = np.linalg.norm(encoding)
            else: raise ValueError(
                f"Unsupported compression method: {compression_method}")
        return summary, encoding

    if not isinstance(data, pd.DataFrame):
        if force: 
            data = build_data_if ( data, force=force, to_frame=True, 
                                  input_name="sum_col_", raise_warning="mute")
        else:
            raise TypeError("The input data must be a pandas DataFrame."
                            " Or set force to 'True' to build a temporary"
                            " dataFrame.")
    
    for column_name in column_names:
        if column_name not in data.columns:
            raise ValueError(f"The column {column_name} does not exist in the DataFrame.")
        if not pd.api.types.is_string_dtype(data[column_name]):
            raise TypeError(f"The column {column_name} must be of type str.")

        summarized_and_encoded = data[column_name].apply(
            lambda text: _summarize_and_encode(text) if isinstance(
                text, str) and text else (text, None))
        data[column_name] = summarized_and_encoded.apply(lambda x: x[0])
        
        if encode:
            encoded_column_name = f"{column_name}_encoded"
            data[encoded_column_name] = summarized_and_encoded.apply(
                lambda x: x[1] if x[1] is not None else None)

    if drop_original:
        data.drop(columns=column_names, inplace=True)
        data.columns = [ c.replace ("_encoded", '') for c in data.columns]

    return data

def simple_extractive_summary(
        texts: List[str], raise_exception: bool = True, encode: bool = False
    ) -> Union[str, Tuple[str, ArrayLike]]:
    """
    Generates a simple extractive summary from a list of texts. 
    
    Function selects the sentence with the highest term frequency-inverse 
    document frequency (TF-IDF) score and optionally returns its TF-IDF 
    encoding.

    Parameters
    ----------
    texts : List[str]
        A list where each element is a string representing a sentence or 
        passage of text.
    raise_exception : bool, default True
        Raise ValueError if the input list contains only one sentence.
    encode : bool, default False
        If True, returns the TF-IDF encoding of the most representative sentence 
        along with the sentence.

    Returns
    -------
    Union[str, Tuple[str, np.ndarray]]
        The sentence from the input list that has the highest TF-IDF score.
        If 'encode' is True, also returns the sentence's TF-IDF encoded vector.

    Raises
    ------
    ValueError
        If the input list contains less than two sentences and raise_exception is True.

    Examples
    --------
    >>> from gofast.tools.baseutils import simple_extractive_summary
    >>> messages = [
    ...     "Further explain the background and rationale for the study. "
    ...     "Explain DNA in simple terms for non-scientists. "
    ...     "Explain the objectives of the study which do not seem perceptible. THANKS",
    ...     "We think this investigation is a good thing. In our opinion, it already allows the "
    ...     "initiators to have an idea of what the populations think of the use of DNA in forensic "
    ...     "investigations in Burkina Faso. And above all, know, through this survey, if these "
    ...     "populations approve of the establishment of a possible genetic database in our country."
    ... ]
    >>> summary, encoding = simple_extractive_summary(messages, encode=True)
    >>> print(summary)
    >>> print(encoding) # encoding is the TF-IDF vector of the summary
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if len(texts) < 2:
        if not raise_exception:
            return (texts[0], None) if encode else texts[0]
        raise ValueError("The input list must contain at least two "
                         "sentences for summarization.")

    vectorizer = TfidfVectorizer(stop_words='english')
    # Create a TF-IDF Vectorizer, excluding common English stop words
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Calculate similarity with the entire set of documents
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix).flatten()
    # Exclude the last entry (similarity of the document with itself)
    sorted_idx = np.argsort(similarity_matrix, axis=0)[:-1]
    # The index of the most similar sentence
    most_similar_idx = sorted_idx[-1]
    
    if encode:
        return texts[most_similar_idx], tfidf_matrix[most_similar_idx].todense()

    return texts[most_similar_idx]

def format_long_column_names(
    data:DataFrame, /,  
    max_length:int=10, 
    return_mapping:bool=False, 
    name_case:str='none'
    )->TypeGuard[DataFrame]:
    """
    Modifies long column names in a DataFrame to a more concise format
    
    Function changes the case as specified, and optionally returns a mapping 
    of new column names to original names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with potentially long column names.
    max_length : int, optional
        Maximum length of the formatted column names. Default is 10.
    return_mapping : bool, optional
        If True, returns a mapping of new column names to original names.
    name_case : {'capitalize', 'lowercase', 'none'}, optional
        Case transformation to apply to the new column names. Default is 'none'.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        The DataFrame with modified column names. If return_mapping is True,
        also returns a dictionary mapping new column names to original names.

    Example
    -------
    >>> from gofast.tools.baseutils import format_long_column_names
    >>> data = {'VeryLongColumnNameIndeed': [1, 2, 3], 'AnotherLongColumnName': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> new_df, mapping = format_long_column_names(
        df, max_length=10, return_mapping=True, name_case='capitalize')
    >>> print(new_df.columns)
    >>> print(mapping)
    """
    if not isinstance ( data, pd.DataFrame): 
        raise TypeError ("Input data is not a pandas DataFrame.")
    new_names = {}
    for col in data.columns:
        # Create a formatted name
        new_name = col[:max_length].rstrip('_')

        # Apply case transformation
        if name_case == 'capitalize':
            new_name = new_name.capitalize()
        elif name_case == 'lowercase':
            new_name = new_name.lower()

        new_names[new_name] = col
        data.rename(columns={col: new_name}, inplace=True)

    if return_mapping:
        return data, new_names

    return data

def enrich_data_spectrum(
    data:DataFrame, /, 
    noise_level:float=0.01, 
    resample_size:int=100, 
    synthetic_size:int=100, 
    bootstrap_size:int=100
    )->TypeGuard[DataFrame]:
    """
    Augment a regression dataset using various techniques. 
    
    The technique including adding noise, resampling, generating 
    synthetic data, and bootstrapping.

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame with regression data. Assumes numerical features.
    noise_level : float, optional
        The level of Gaussian noise to add, specified as a fraction of the standard
        deviation of each numerical feature (default is 0.01).
    resample_size : int, optional
        Number of data points to generate via resampling without replacement from the
        original dataset (default is 100).
    synthetic_size : int, optional
        Number of synthetic data points to generate through interpolation between
        pairs of existing data points (default is 100).
    bootstrap_size : int, optional
        Number of data points to generate via bootstrapping (sampling with replacement)
        from the original dataset (default is 100).
    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with the original and newly generated data points.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from gofast.tools.baseutils import enrich_data_spectrum
    >>> boston = load_boston()
    >>> data = pd.DataFrame(boston.data, columns=boston.feature_names)
    >>> augmented_data = enrich_data_spectrum(
        data, noise_level=0.02, resample_size=50, synthetic_size=50, bootstrap_size=50)
    >>> print(augmented_data.shape)
    
    Note 
    ------ 
    `enrich_data_spectrum` proposes several techniques that can be used to 
    augment data in regression as explained: 
        
    - Adding Noise: Adds Gaussian noise to each numerical feature based on a 
      specified noise level relative to the feature's standard deviation.
    - Data Resampling: Randomly resamples (without replacement) a specified 
      number of data points from the original dataset.
    - Synthetic Data Generation: Creates new data points by linear interpolation
      between randomly chosen pairs of existing data points.
    - Bootstrapping: Resamples (with replacement) a specified number of data 
      points, allowing for duplicates.
    
    Adjust the parameters like noise_level, resample_size, synthetic_size, 
    and bootstrap_size according to your dataset and needs.
    This function is quite general and may need to be tailored to fit the 
    specific characteristics of your dataset and regression task.
    Be cautious with the synthetic data generation step; ensure that the 
    generated points are plausible within your problem domain.
    Always validate the model performance with and without the augmented 
    data to ensure that the augmentation is beneficial.
    
    """
    from sklearn.utils import resample
    data = build_data_if(data,  to_frame=True, force=True, 
                         input_name="feature_",  raise_warning='mute')
    data = to_numeric_dtypes( data , pop_cat_features= True) 
    if len(data.columns)==0: 
        raise TypeError("Numeric features are expected.")
    augmented_df = data.copy()

    # Adding noise
    for col in data.select_dtypes(include=[np.number]):
        noise = np.random.normal(0, noise_level * data[col].std(),
                                 size=data.shape[0])
        augmented_df[col] += noise

    # Data resampling
    resampled_data = resample(data, n_samples=resample_size, replace=False)
    augmented_df = pd.concat([augmented_df, resampled_data], axis=0)

    # Synthetic data generation
    for _ in range(synthetic_size):
        idx1, idx2 = np.random.choice(data.index, 2, replace=False)
        synthetic_point = data.loc[idx1] + np.random.rand() * (
            data.loc[idx2] - data.loc[idx1])
        # Create a DataFrame from synthetic_point to use with pd.concat
        synthetic_df = pd.DataFrame([synthetic_point], columns=data.columns)
        augmented_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
        # append deprecated
        # augmented_df = augmented_df.append(synthetic_point, ignore_index=True)

    # Bootstrapping
    bootstrapped_data = resample(data, n_samples=bootstrap_size, replace=True)
    augmented_df = pd.concat([augmented_df, bootstrapped_data], axis=0)

    return augmented_df.reset_index(drop=True)

def sanitize(
    data:DataFrame, /, 
    fill_missing:Optional[str]=None, 
    remove_duplicates:bool=True, 
    outlier_method:Optional[str]=None, 
    consistency_transform:Optional[str]=None, 
    threshold:float|int=3
    )->TypeGuard[DataFrame]:
    """
    Perform data cleaning on a DataFrame with many options. 
    
    Options consists for handling missing values, removing duplicates, 
    detecting and removing outliers, and transforming string data for 
    consistency.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cleaned.
    fill_missing : {'median', 'mean', 'mode', None}, optional
        Method to fill missing values. If None, no filling is performed.
        - 'median': Fill missing values with the median of each column.
        - 'mean': Fill missing values with the mean of each column.
        - 'mode': Fill missing values with the mode of each column.
        Suitable for datasets where missing values are present and a simple 
        imputation is required.
    remove_duplicates : bool, optional
        If True, removes duplicate rows from the DataFrame. Useful in scenarios
        where duplicate entries 
        do not provide additional information and might skew the analysis.
    outlier_method : {'z_score', 'iqr', None}, optional
        Method for outlier detection and removal. If None, no outlier 
        processing is performed.
        - 'z_score': Identifies and removes outliers using Z-score.
        - 'iqr': Identifies and removes outliers using the Interquartile Range.
        Choose based on the nature of the data and the requirement of the analysis.
    consistency_transform : {'lower', 'upper', None}, optional
        Transformation to apply to string columns for consistency. If None,
        no transformation is applied.
        - 'lower': Converts strings to lowercase.
        - 'upper': Converts strings to uppercase.
        Useful for categorical data where case consistency is important.
    threshold : float, optional
        The threshold value used for outlier detection methods. Default is 3.
        For 'z_score', it represents
        the number of standard deviations from the mean. For 'iqr', it is the 
        multiplier for the IQR.

    Returns
    -------
    pd.DataFrame
        The cleaned and processed DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.baseutils import clean_data
    >>> data = {'A': [1, 2, None, 4], 'B': ['X', 'Y', 'Y', None], 'C': [1, 1, 2, 2]}
    >>> df = pd.DataFrame(data)
    >>> cleaned_df = clean_data(df, fill_missing='median', remove_duplicates=True,
                                outlier_method='z_score', 
                                consistency_transform='lower', threshold=3)
    >>> print(cleaned_df)
    """
    data = build_data_if(data, to_frame=True, force=True, 
                         input_name="feature_",  raise_warning='mute')
    data = to_numeric_dtypes( data ) # verify integrity 
    df_cleaned = data.copy()
    if fill_missing:
        fill_methods = {
            'median': data.median(),
            'mean': data.mean(),
            'mode': data.mode().iloc[0]
        }
        df_cleaned.fillna(fill_methods.get(fill_missing, None), inplace=True)

    if remove_duplicates:
        df_cleaned.drop_duplicates(inplace=True)

    if outlier_method:
        if outlier_method == 'z_score':
            for col in df_cleaned.select_dtypes(include=[np.number]):
                df_cleaned = df_cleaned[(np.abs(df_cleaned[col] - df_cleaned[col].mean()
                                                ) / df_cleaned[col].std()) < threshold]
        elif outlier_method == 'iqr':
            for col in df_cleaned.select_dtypes(include=[np.number]):
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                df_cleaned = df_cleaned[~((df_cleaned[col] < (
                    Q1 - threshold * IQR)) | (df_cleaned[col] > (Q3 + threshold * IQR)))]

    if consistency_transform:
        transform_methods = {
            'lower': lambda x: x.lower(),
            'upper': lambda x: x.upper()
        }
        for col in df_cleaned.select_dtypes(include=[object]):
            df_cleaned[col] = df_cleaned[col].astype(str).map(
                transform_methods.get(consistency_transform, lambda x: x))

    return df_cleaned

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

def read_data (
    f: str|pathlib.PurePath, 
    sanitize: bool= ..., 
    reset_index: bool=..., 
    comments: str="#", 
    delimiter: str=None, 
    columns: List[str]=None,
    npz_objkey: str= None, 
    verbose: bool= ..., 
    **read_kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a data frame.  
    
    Parameters 
    -----------
    f: str, Path-like object 
        File path or Pathlib object. Must contain a valid file name  and 
        should be a readable file or url 
        
    sanitize: bool, default=False, 
        Push a minimum sanitization of the data such as: 
        - replace a non-alphabetic column items with a pattern '_' 
        - cast data values to numeric if applicable 
        - drop full NaN columns and rows in the data 
           
    reset_index: bool, default=False, 
        Reset index if full NaN columns are dropped after sanitization. 
        Apply minimum data sanitization after reading data. 
     
    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. 

    delimiter: str, optional
       The character used to separate the values. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. The default is whitespace.

    npz_objkey: str, optional 
       Dataset key to indentify array in multiples array storages in '.npz' 
       format.  If key is not set during 'npz' storage, ``arr_0`` should 
       be used.Capable to read text and numpy formats ('.npy' and '.npz') data. 
       Note that when data is stored in compressed ".npz" format, provided the 
        '.npz' object key  as argument of parameter `npz_objkey`. If None, 
        only the first array should be read and ``npz_objkey='arr_0'``. 
          
    verbose: bool, default=0 
       Outputs message for user guide. 
       
    read_kws: dict, 
       Additional keywords arguments passed to pandas readable file keywords. 
        
    Returns 
    -------
    f: :class:`pandas.DataFrame` 
        A dataframe with head contents by default.  
        
    See Also 
    ---------
    np.loadtxt: 
        load text file.  
    np.load 
       Load uncompressed or compressed numpy `.npy` and `.npz` formats. 
    gofast.tools.baseutils.save_or_load: 
        Save or load numpy arrays.
       
    """
    def min_sanitizer ( d, /):
        """ Apply a minimum sanitization to the data `d`."""
        return to_numeric_dtypes(
            d, sanitize_columns= True, 
            drop_nan_columns= True, 
            reset_index=reset_index, 
            verbose = verbose , 
            fill_pattern='_', 
            drop_index = True
            )
    sanitize, reset_index, verbose = ellipsis2false (
        sanitize, reset_index, verbose )
    if ( isinstance ( f, str ) 
            and str(os.path.splitext(f)[1]).lower()in (
                '.txt', '.npy', '.npz')
            ): 
        f = save_or_load(f, task = 'load', comments=comments, 
                         delimiter=delimiter )
        # if extension is .npz
        if isinstance(f, np.lib.npyio.NpzFile):
            npz_objkey = npz_objkey or "arr_0"
            f = f[npz_objkey] 

        if columns is not None: 
            columns = is_iterable(columns, exclude_string= True, 
                                  transform =True, parse_string =True 
                                  )
            if len( columns )!= f.shape [1]: 
                warnings.warn(f"Columns expect {f.shape[1]} attributes."
                              f" Got {len(columns)}")
            
        f = pd.DataFrame(f, columns=columns )
        
    if isinstance (f, pd.DataFrame): 
        if sanitize: 
            f = min_sanitizer (f)
        return  f 
    
    cpObj= Config().parsers 
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} files"
                        )
    try : 
        f = cpObj[ex](f, **read_kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except BaseException as e : 
        raise FileHandlingError (
            f"Cannot parse the file : {os.path.basename (f)!r}. "+  str(e))
    if sanitize: 
        f = min_sanitizer (f)
        
    return f 
    
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

def lowertify(
    *values, strip: bool = True, return_origin: bool = False, 
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
 
def request_data(
    url: str, 
    method: str = 'get',
    data: Optional[Any] = None, 
    as_json: bool = ..., 
    as_text: bool = ..., 
    stream: bool = ..., 
    raise_status: bool = ..., 
    save_to_file: bool = ..., 
    filename: Optional[str] = None, 
    show_progress: bool = ...,
    **kwargs
) -> Union[str, dict, ...]:
    """
    Perform an HTTP request to a specified URL and process the response, with 
    optional progress bar visualization.

    Parameters
    ----------
    url : str
        The URL to which the HTTP request is sent.
    method : str, optional
        The HTTP method to use for the request. Supported values are 'get' 
        and 'post'. Default is 'get'.
    data : Any, optional
        The data to send in the body of the request, used with 'post' method.
    as_json : bool, optional
        If True, parses the response as JSON. Default is False.
    as_text : bool, optional
        If True, returns the response as a string. Default is False.
    stream : bool, optional
        If True, streams the response. Useful for large file downloads.
        Default is False.
    raise_status : bool, optional
        If True, raises an HTTPError for bad HTTP responses. 
        Default is False.
    save_to_file : bool, optional
        If True, saves the response content to a file. Default is False.
    filename : str, optional
        File path for saving response content. Required if 
        `save_to_file` is True.
    show_progress : bool, optional
        If True, displays a progress bar during file download. 
        Default is False.
    **kwargs
        Additional keyword arguments passed to the requests method
        (e.g., headers, cookies).

    Returns
    -------
    Union[str, dict, requests.Response]
        The server's response. Depending on the flags, this can be a string,
        a dictionary, or a raw Response object.

    Raises
    ------
    ValueError
        If `save_to_file` is True but no `filename` is provided.
        If an invalid HTTP method is specified.

    Examples
    --------
    >>> from gofast.tools.baseutils import request_data
    >>> response = request_data('https://api.github.com/user',
                                auth=('user', 'pass'), as_json=True)
    >>> print(response)
    """
    import_optional_dependency('requests' ) 
    import requests 
    
    (as_text, as_json, stream, raise_status, save_to_file,
     show_progress) = ellipsis2false(
        as_text, as_json,  stream, raise_status , save_to_file,
        show_progress)
    
    if save_to_file and not filename:
        raise ValueError("A filename must be provided when "
                         "'save_to_file' is True.")

    request_method = getattr(requests, method.lower(), None)
    if not request_method:
        raise ValueError(f"Invalid HTTP method: {method}")

    response = request_method(url, data=data, stream=stream, **kwargs)

    if save_to_file:
        with open(filename, 'wb') as file:
            if show_progress:
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='iB',ascii=True,
                                    unit_scale=True, ncols=97)
            for chunk in response.iter_content(chunk_size=1024):
                if show_progress:
                    progress_bar.update(len(chunk))
                file.write(chunk)
            if show_progress:
                progress_bar.close()

    if raise_status:
        response.raise_for_status()

    return response.text if as_text else ( 
        response.json () if as_json else response )

@deprecated("Deprecated function. Should be remove next release."
            "Use `gofast.tools.fetch_remote_data` instead.")
def get_remote_data(
    remote_file: str, 
    save_path: Optional[str] = None, 
    raise_exception: bool = True
) -> bool:
    """
    Retrieve data from a remote location and optionally save it to a 
    specified path.

    Parameters
    ----------
    remote_file : str
        The full path URL to the remote file to be downloaded.
    
    save_path : str, optional
        The local file system path where the downloaded file should be saved.
        If None, the file is saved in the current directory. Default is None.
    
    raise_exception : bool, default True
        If True, raises a ConnectionRefusedError when the connection fails.
        Otherwise, prints the error message.

    Returns
    -------
    bool
        True if the file was successfully downloaded; False otherwise.

    Raises
    ------
    ConnectionRefusedError
        If the connection fails and `raise_exception` is True.

    Examples
    --------
    >>> from gofast.tools.baseutils import get_remote_data
    >>> status = get_remote_data('https://example.com/file.csv', save_path='/local/path')
    >>> print(status)
    """

    connect_reason = (
        "ConnectionRefusedError: Failed to connect to the remote server. "
        "Possible reasons include:\n"
        "1. The server is not running, thus not listening to the port.\n"
        "2. The server is running, but the port is blocked by a firewall.\n"
        "3. A security program on the PC is blocking several ports."
    )
    validate_url(remote_file)
    print(f"---> Fetching {remote_file!r}...")

    try:
        # Setting up the progress bar
        with tqdm(total=3, ascii=True, desc=f'Fetching {os.path.basename(remote_file)}', 
                  ncols=97) as pbar:
            _ , rfile = os.path.dirname(remote_file), os.path.basename(remote_file)
            status = False

            for k in range(3):
                try:
                    response = urllib.request.urlopen(remote_file)
                    data = response.read() # a `bytes` object

                    # Save the data to file
                    with open(rfile, 'wb') as out_file:
                        out_file.write(data)
                    status = True
                    break
                except TimeoutError:
                    if k == 2:
                        print("---> Connection timed out.")
                except Exception as e:
                    print(f"---> An error occurred: {e}")
                finally:
                    pbar.update(1)

            if status:
                # Move the file to the specified save_path
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    shutil.move(os.path.realpath(rfile),
                                os.path.join(save_path, rfile))
            else:
                print(f"\n---> Failed to download {remote_file!r}.")
                if raise_exception:
                    raise ConnectionRefusedError(connect_reason)

            return status

    except Exception as e:
        print(f"An error occurred during the download: {e}")
        if raise_exception:
            raise e
        return False

def fetch_remote_data(
    remote_file_url: str, 
    save_path: Optional[str] = None, 
    raise_exception: bool = True
 ) -> bool:
    """
    Download a file from a remote URL and optionally save it to a specified location.

    This function attempts to download a file from the given URL. If `save_path` is 
    provided, it saves the file to that location, otherwise, it saves it in the 
    current working directory. If the download fails, it can optionally raise an 
    exception or return False.

    Parameters
    ----------
    remote_file_url : str
        The URL of the remote file to be downloaded.
    save_path : str, optional
        The local directory path where the downloaded file should be saved. 
        If None, the file is saved in the current directory. Default is None.
    raise_exception : bool, default True
        If True, raises an exception upon failure. Otherwise, returns False.

    Returns
    -------
    bool
        True if the file was successfully downloaded, False otherwise.

    Raises
    ------
    ConnectionRefusedError
        If the download fails and `raise_exception` is True.

    Examples
    --------
    >>> status = get_remote_data('https://example.com/file.csv', save_path='/local/path')
    >>> print(status)

    """
    def handle_download_error(e: Exception, message: str) -> None:
        """
        Handle download errors, either by raising an exception or printing 
        an error message.

        Parameters
        ----------
        e : Exception
            The exception that was raised.
        message : str
            The error message to be printed or included in the raised exception.

        Raises
        ------
        Exception
            The original exception, if `raise_exception` is True.
        """
        print(message)
        if raise_exception:
            raise e

    def move_file_to_save_path(file_name: str) -> None:
        """
        Move the downloaded file to the specified save path.

        Parameters
        ----------
        file_name : str
            The name of the file to be moved.
        """
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            shutil.move(os.path.realpath(file_name), os.path.join(
                save_path, file_name))

    try:
        file_name = os.path.basename(remote_file_url)
        print(f"---> Fetching '{remote_file_url}'...")

        with tqdm(total=3, ascii=True, desc=f'Fetching {file_name}',
                  ncols=97) as progress_bar:
            for attempt in range(3):
                try:
                    response = urllib.request.urlopen(remote_file_url)
                    data = response.read()

                    with open(file_name, 'wb') as file:
                        file.write(data)

                    move_file_to_save_path(file_name)
                    return True

                except TimeoutError:
                    if attempt == 2:
                        handle_download_error(
                            TimeoutError(), "Connection timed out while"
                            f" downloading '{remote_file_url}'.")
                except Exception as e:
                    handle_download_error(
                        e, f"An error occurred while downloading '{remote_file_url}': {e}")
                finally:
                    progress_bar.update(1)

            # If all attempts fail
            return False

    except Exception as e:
        handle_download_error(e, f"An unexpected error occurred during the download: {e}")
        return False

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
        import_optional_dependency("tqdm")
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
    import_optional_dependency("importlib")
    import importlib.resources as pkg_resources
    return pkg_resources.is_resource(package, resource)

def _is_readable (
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
    
def scrape_web_data(
    url, element, 
    class_name=None
    ):
    """
    Scrape data from a web page.

    Parameters
    ----------
    url : str
        The URL of the web page to scrape.

    element : str
        The HTML element to search for.

    class_name : str, optional
        The class attribute of the HTML element to 
        narrow down the search.
    
    Returns
    -------
    list of bs4.element.Tag
        A list of BeautifulSoup Tag objects that match the search query.

    Examples
    --------
    >>> url = 'https://example.com'
    >>> element = 'div'
    >>> class_name = 'content'
    >>> data = scrape_web_data(url, element, class_name)
    >>> for item in data:
    >>>     print(item.text)
    # Assuming the function scrape_web_data is defined as above.

    >>> url = 'https://example.com/articles'
    >>> element = 'h1'
    >>> data = scrape_web_data(url, element)

    >>> for header in data:
           print(header.text)  # prints the text of each <h1> tag

    """
    
    import_optional_dependency('requests' ) 
    extra= (" Needs `BeautifulSoup` from `bs4` package" )
    import_optional_dependency('bs4', extra = extra  ) 
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        if class_name:
            elements = soup.find_all(element, class_=class_name)
        else:
            elements = soup.find_all(element)
        return elements
    else:
        response.raise_for_status()  
     
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
    
def run_shell_command(command, progress_bar_duration=30):
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

    Returns:
    --------
    None
    
    Example 
    -------
    >>> from gofast.tools.baseutils import run_shell_command 
    >>> run_with_progress_bar(["pip", "install", "gofast"])
    """
    def run_command(command):
        subprocess.run(command, check=True)

    def show_progress_bar(duration):
        with tqdm(total=duration, desc="Installing", 
                  bar_format="{l_bar}{bar}", ncols=100, ascii=True)  as pbar:
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

def handle_datasets_in_h5(
    file_path: str,
    datasets: Optional[Dict[str, ArrayLike]] = None, 
    operation: str = 'store'
    ) -> Union[None, Dict[str, ArrayLike]]:
    """
    Handles storing or retrieving multiple datasets in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the 
        datasets (numpy arrays).
        Required if operation is 'store'. Default is None.
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    dict or None
        If operation is 'retrieve', returns a dictionary where keys are dataset
        names and values are the datasets (numpy arrays).
        If operation is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified.
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    Storing datasets:
    >>> data1 = np.random.rand(100, 10)
    >>> data2 = np.random.rand(200, 5)
    >>> handle_datasets_in_h5('my_datasets.h5', 
                              {'dataset1': data1, 'dataset2': data2}, operation='store')

    Retrieving datasets:
    >>> datasets = handle_datasets_in_h5('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with h5py.File(file_path, 'w') as h5file:
            for name, data in datasets.items():
                h5file.create_dataset(name, data=data)

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with h5py.File(file_path, 'r') as h5file:
            for name in h5file.keys():
                datasets_retrieved[name] = h5file[name][...]
                
        return datasets_retrieved

def handle_datasets_with_hdfstore(
    file_path: str, 
    datasets: Optional[Dict[str, DataFrame]] = None, 
    operation: str = 'store') -> Union[None, Dict[str, DataFrame]]:
    """
    Handles storing or retrieving multiple Pandas DataFrames in an HDF5 
    file using pd.HDFStore.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the datasets
        (Pandas DataFrames).
        Required if operation is 'store'. Default is None.
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    dict or None
        If operation is 'retrieve', returns a dictionary where keys are dataset 
        names and values are the datasets (Pandas DataFrames).
        If operation is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified.
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import handle_datasets_with_hdfstore
    
    Storing datasets:
    >>> df1 = pd.DataFrame(np.random.rand(100, 10), columns=[f'col_{i}' for i in range(10)])
    >>> df2 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    >>> handle_datasets_with_hdfstore('my_datasets.h5', {'df1': df1, 'df2': df2}, operation='store')

    Retrieving datasets:
    >>> datasets = handle_datasets_with_hdfstore('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with pd.HDFStore(file_path, 'w') as store:
            for name, df in datasets.items():
                store.put(name, df)

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with pd.HDFStore(file_path, 'r') as store:
            for name in store.keys():
                datasets_retrieved[name.strip('/')] = store[name]
        return datasets_retrieved
    
def store_or_retrieve_data(
    file_path: str,
    datasets: Optional[Dict[str, Union[ArrayLike, DataFrame]]] = None,
    operation: str = 'store'
) -> Optional[Dict[str, Union[ArrayLike, DataFrame]]]:
    """
    Handles storing or retrieving multiple datasets (numpy arrays or Pandas
    DataFrames) in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file for storing or retrieving datasets.
    datasets : dict, optional
        A dictionary with dataset names as keys and datasets 
        (numpy arrays or Pandas DataFrames) as values.
        Required if operation is 'store'.
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]]
        If operation is 'retrieve', returns a dictionary with dataset names 
        as keys and datasets as values. If operation is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified or required parameters are missing.
    TypeError
        If provided datasets are not in supported formats 
        (numpy arrays or pandas DataFrames).
    """

    valid_operations = {'store', 'retrieve'}
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation '{operation}'. "
                         f"Choose from {valid_operations}.")

    with pd.HDFStore(file_path, mode='a' if operation == 'store' else 'r') as store:
        if operation == 'store':
            if not datasets:
                raise ValueError("Datasets are required for the 'store' operation.")

            for name, data in datasets.items():
                if not isinstance(data, (pd.DataFrame, np.ndarray)):
                    raise TypeError("Unsupported data type. Only numpy arrays "
                                    "and pandas DataFrames are supported.")
                
                store[name] = pd.DataFrame(data) if isinstance(data, np.ndarray) else data

        elif operation == 'retrieve':
            return {name.replace ("/", ""): store[name] for name in store.keys()}
        
def base_storage(
    file_path: str,
    datasets: Optional[Dict[str, Union[ArrayLike, DataFrame]]] = None, 
    operation: str = 'store'
) -> Union[None, Dict[str, Union[ArrayLike, DataFrame]]]:
    """
    Handles storing or retrieving multiple datasets (numpy arrays or Pandas 
    DataFrames) in an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file where datasets will be stored or from which 
        datasets will be retrieved.
    datasets : dict, optional
        A dictionary where keys are dataset names and values are the 
        datasets (numpy arrays or Pandas DataFrames).
        Required if operation is 'store'. 
    operation : str
        The operation to perform - 'store' for storing datasets, 'retrieve' 
        for retrieving datasets.

    Returns
    -------
    dict or None
        If operation is 'retrieve', returns a dictionary where keys are dataset
        names and values are the datasets (numpy arrays or Pandas DataFrames).
        If operation is 'store', returns None.

    Raises
    ------
    ValueError
        If an invalid operation is specified.
    OSError
        If the file cannot be opened or created.

    Examples
    --------
    Storing datasets:
    >>> data1 = np.random.rand(100, 10)
    >>> df1 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)),
                           columns=['A', 'B', 'C', 'D', 'E'])
    >>> store_data('my_datasets.h5', {'dataset1': data1, 'df1': df1},
                              operation='store')

    Retrieving datasets:
    >>> datasets = store_data('my_datasets.h5', operation='retrieve')
    >>> print(datasets.keys())
    """
    if operation not in ['store', 'retrieve']:
        raise ValueError("Invalid operation. Please choose 'store' or 'retrieve'.")

    if operation == 'store':
        if datasets is None:
            raise ValueError("Datasets parameter is required for storing data.")

        with h5py.File(file_path, 'w') as h5file:
            for name, data in datasets.items():
                if isinstance(data, pd.DataFrame):
                    data.to_hdf(file_path, key=name, mode='a')
                elif isinstance(data, np.ndarray):
                    h5file.create_dataset(name, data=data)
                else:
                    raise TypeError("Unsupported data type. Only numpy arrays "
                                    "and pandas DataFrames are supported.")

    elif operation == 'retrieve':
        datasets_retrieved = {}
        with h5py.File(file_path, 'r') as h5file:
            for name in h5file.keys():
                try:
                    datasets_retrieved[name] = pd.read_hdf(file_path, key=name)
                except (KeyError, TypeError):
                    datasets_retrieved[name] = h5file[name][...]

        return datasets_retrieved
    
def verify_data_integrity(data: DataFrame, /) -> Tuple[bool, dict]:
    """
    Verifies the integrity of data within a DataFrame. 
    
    Data integrity checks are crucial in data analysis and machine learning 
    to ensure the reliability and correctness of any conclusions drawn from 
    the data. This function performs several checks including looking for 
    missing values, detecting duplicates, and identifying outliers, which
    are common issues that can lead to misleading analysis or model training 
    results.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to verify.

    Returns
    -------
    Tuple[bool, dict]
        A tuple containing:
        - A boolean indicating if the data passed all integrity checks 
          (True if no issues are found).
        - A dictionary with the details of the checks, including counts of 
        missing values, duplicates, and outliers by column.

    Example
    -------
    >>> data = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6], 'C': [7, 8, 8]})
    >>> is_valid, report = verify_data_integrity(data)
    >>> print(f"Data is valid: {is_valid}\nReport: {report}")

    Notes
    -----
    Checking for missing values is essential as they can indicate data 
    collection issues or errors in data processing. Identifying duplicates is 
    important to prevent skewed analysis results, especially in cases where 
    data should be unique (e.g., unique user IDs). Outlier detection is 
    critical in identifying data points that are significantly different from 
    the majority of the data, which might indicate data entry errors or other 
    anomalies.
    
    - The method used for outlier detection in this function is the 
      Interquartile Range (IQR) method. It's a simple approach that may not be
      suitable for all datasets, especially those with non-normal distributions 
      or where more sophisticated methods are required.
    - The function does not modify the original DataFrame.
    """
    report = {}
    is_valid = True
    # check whether dataframe is passed
    is_frame (data, df_only=True, raise_exception=True )
    data = to_numeric_dtypes(data)
    # Check for missing values
    missing_values = data.isnull().sum()
    report['missing_values'] = missing_values
    if missing_values.any():
        is_valid = False

    # Check for duplicates
    duplicates = data.duplicated().sum()
    report['duplicates'] = duplicates
    if duplicates > 0:
        is_valid = False

    # Check for potential outliers
    outlier_report = {}
    for col in data.select_dtypes(include=['number']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_report[col] = len(outliers)
        if len(outliers) > 0:
            is_valid = False

    report['outliers'] = outlier_report

    return is_valid, report

def audit_data(
    data: DataFrame,/,  
    dropna_threshold: float = 0.5, 
    categorical_threshold: int = 10, 
    handle_outliers: bool = False,
    handle_missing: bool = True, 
    handle_scaling: bool = False, 
    handle_date_features: bool = False, 
    handle_categorical: bool = False, 
    replace_with: str = 'median', 
    lower_quantile: float = 0.01, 
    upper_quantile: float = 0.99,
    fill_value: Optional[Any] = None,
    scale_method: str = "minmax",
    missing_method: str = 'drop_cols', 
    outliers_method: str = "clip", 
    date_features: Optional[List[str]] = None,
    day_of_week: bool = False, 
    quarter: bool = False, 
    format_date: Optional[str] = None, 
    return_report: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Audits and preprocesses a DataFrame for analytical consistency. 
    
    This function streamlines the data cleaning process by handling various 
    aspects of data quality, such as outliers, missing values, and data scaling. 
    It provides flexibility to choose specific preprocessing steps according 
    to the needs of the analysis or modeling.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be audited and preprocessed. It should be a pandas 
        DataFrame containing the data to be cleaned.

    dropna_threshold : float, optional
        Specifies the threshold for dropping columns or rows with missing 
        values. It determines the proportion of missing values above which 
        a column or row will be dropped from the DataFrame. 
        The default value is 0.5 (50%).

    categorical_threshold : int, optional
        Defines the maximum number of unique values a column can have to be 
        considered as a categorical variable. Columns with unique values 
        less than or equal to this threshold will be converted to categorical
        type.

    handle_outliers : bool, optional
        Determines whether to apply outlier handling on numerical columns. 
        If set to True, outliers in the data will be addressed according 
        to the specified method.

    handle_missing : bool, optional
        If True, the function will handle missing data in the DataFrame based 
        on the specified missing data handling method.

    handle_scaling : bool, optional
        Indicates whether to scale numerical columns using the specified 
        scaling method. Scaling is essential for certain analyses and modeling 
        techniques, especially when variables are on different scales.

    handle_date_features : bool, optional
        If True, columns specified as date features will be converted to 
        datetime format, and additional date-related features 
        (like day of the week, quarter) will be extracted.

    handle_categorical : bool, optional
        Enables the handling of categorical features. If set to True, 
        numerical columns with a number of unique values below the categorical
        threshold will be treated as categorical.

    replace_with : str, optional
        For outlier handling, specifies the method of replacement 
        ('mean' or 'median') for the 'replace' outlier method. It determines 
        how outliers will be replaced in the dataset.

    lower_quantile : float, optional
        The lower quantile value used for clipping outliers. It sets the 
        lower boundary for outlier detection and handling.

    upper_quantile : float, optional
        The upper quantile value for clipping outliers. It sets the upper 
        boundary for outlier detection and handling.

    fill_value : Any, optional
        Specifies the value to be used for filling missing data when the 
        missing data handling method is set to 'fill_value'.

    scale_method : str, optional
        Determines the method for scaling numerical data. Options include 
        'minmax' (scales data to a range of [0, 1]) and 'standard' 
        (scales data to have zero mean and unit variance).

    missing_method : str, optional
        The method used to handle missing data in the DataFrame. Options 
        include 'drop_cols' (drop columns with missing data) and other 
        methods based on specified criteria such as 'drop_rows', 'fill_mean',
        'fill_median', 'fill_value'. 

    outliers_method : str, optional
        The method used for handling outliers in the dataset. Options 
        include 'clip' (limits the extreme values to specified quantiles) and 
        other outlier handling methods such as 'remove' and 'replace'.

    date_features : List[str], optional
        A list of column names in the DataFrame to be treated as date features. 
        These columns will be converted to datetime and additional date-related
        features will be extracted.

    day_of_week : bool, optional
        If True, adds a column representing the day of the week for each 
        date feature column.

    quarter : bool, optional
        If True, adds a column representing the quarter of the year for 
        each date feature column.

    format_date : str, optional
        Specifies the format of the date columns if they are not in standard 
        datetime format.

    return_report : bool, optional
        If True, the function returns a detailed report summarizing the 
        preprocessing steps performed on the DataFrame.

    view : bool, optional
        Enables visualization of the data's state before and after 
        preprocessing. If True, displays comparative heatmaps for each step.

    cmap : str, optional
        The colormap for the heatmap visualizations, enhancing the clarity 
        and aesthetics of the plots.

    fig_size : Tuple[int, int], optional
        Determines the size of the figure for the visualizations, allowing 
        customization of the plot dimensions.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        The audited and preprocessed DataFrame. If return_report is True, 
        also returns a comprehensive report detailing the transformations 
        applied.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import audit_data
    >>> data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    >>> audited_data, report = audit_data(data, handle_outliers=True, return_report=True)
    """
    is_frame (data, df_only=True, raise_exception=True, 
              objname="Data for auditing" )
    report = {}
    data_copy = data.copy()

    def update_report(new_data, step_report):
        nonlocal data, report
        if return_report:
            data, step_report = new_data
            report = {**report, **step_report}
        else:
            data = new_data

    # Handling outliers
    if handle_outliers:
        update_report(handle_outliers_in_data(
            data, method=outliers_method, replace_with=replace_with,
            lower_quantile=assert_ratio(lower_quantile),
            upper_quantile=assert_ratio(upper_quantile),
            return_report=return_report),
            {})

    # Handling missing data
    if handle_missing:
        update_report(handle_missing_data(
            data, method=missing_method, dropna_threshold=assert_ratio(
                dropna_threshold), fill_value=fill_value, 
            return_report=return_report), {})

    # Handling date features
    if handle_date_features and date_features:
        update_report(convert_date_features(
            data, date_features, day_of_week=day_of_week, quarter=quarter, 
            format=format_date, return_report=return_report), {})

    # Scaling data
    if handle_scaling:
        update_report(scale_data(
            data, method=scale_method, return_report=return_report), {})

    # Handling categorical features
    if handle_categorical:
        update_report(handle_categorical_features(
            data, categorical_threshold=categorical_threshold, 
            return_report=return_report), {})

    # Compare initial and final data if view is enabled
    if view:
        plt.figure(figsize=(fig_size[0], fig_size[1] * 2))
        plt.subplot(2, 1, 1)
        sns.heatmap(data_copy.isnull(), yticklabels=False,
                    cbar=False, cmap=cmap)
        plt.title('Data Before Auditing')

        plt.subplot(2, 1, 2)
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False,
                    cmap=cmap)
        plt.title('Data After Auditing')
        plt.show()

    return (data, report) if return_report else data

def handle_categorical_features(
    data: DataFrame, /, 
    categorical_threshold: int = 10,
    return_report: bool = False,
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Converts numerical columns with a limited number of unique values 
    to categorical columns in the DataFrame and optionally visualizes the 
    data distribution before and after the conversion.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process.
    categorical_threshold : int, optional
        Maximum number of unique values in a column for it to be considered 
        categorical.
    return_report : bool, optional
        If True, returns a report summarizing the categorical feature handling.
    view : bool, optional
        If True, displays a heatmap of the data distribution before and after 
        handling.
    cmap : str, optional
        The colormap for the heatmap visualization.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame with categorical features handled and optionally a report.

    Example
    -------
    >>> from gofast.tools.baseutils import handle_categorical_features
    >>> data = pd.DataFrame({'A': [1, 2, 1, 3], 'B': range(10)})
    >>> updated_data, report = handle_categorical_features(
        data, categorical_threshold=3, return_report=True, view=True)
    """
    is_frame (data, df_only=True, raise_exception=True)
    original_data = data.copy()
    report = {'converted_columns': []}
    numeric_cols = data.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if data[col].nunique() <= categorical_threshold:
            data[col] = data[col].astype('category')
            report['converted_columns'].append(col)

    # Visualization of data distribution before and after handling
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[numeric_cols].nunique().to_frame().T, 
                    annot=True, cbar=False, cmap=cmap)
        plt.title('Unique Values Before Categorization')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[numeric_cols].nunique().to_frame().T, annot=True,
                    cbar=False, cmap=cmap)
        plt.title('Unique Values After Categorization')
        plt.show()

    return (data, report) if return_report else data

def convert_date_features(
    data: DataFrame, /, 
    date_features: List[str], 
    day_of_week: bool = False, 
    quarter: bool = False,
    format: Optional[str] = None,
    return_report: bool = False,
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Converts specified columns in the DataFrame to datetime and extracts 
    relevant features. 
    
    Optionally Function returns a report of the transformations and 
    visualizing the data distribution  before and after conversion.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the date columns.
    date_features : List[str]
        List of column names to be converted into datetime and to extract 
        features from.
    day_of_week : bool, optional
        If True, adds a column representing the day of the week. Default is False.
    quarter : bool, optional
        If True, adds a column representing the quarter of the year.
        Default is False.
    format : str, optional
        The specific format of the date columns if they are not in a standard
        datetime format.
    return_report : bool, optional
        If True, returns a report summarizing the date feature transformations.
    view : bool, optional
        If True, displays a comparative heatmap of the data distribution 
        before and after the conversion.
    cmap : str, optional
        The colormap for the heatmap visualization.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame with additional date-related features and optionally a report.

    Example
    -------
    >>> from gofast.tools.baseutils import convert_date_features
    >>> data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02']})
    >>> updated_data, report = convert_date_features(
        data, ['date'], day_of_week=True, quarter=True, return_report=True, view=True)
    """
    is_frame (data, df_only=True, raise_exception=True)
    original_data = data.copy()
    report = {'converted_columns': date_features, 'added_features': []}

    for feature in date_features:
        data[feature] = pd.to_datetime(data[feature], format=format)
        year_col = f'{feature}_year'
        month_col = f'{feature}_month'
        day_col = f'{feature}_day'
        data[year_col] = data[feature].dt.year
        data[month_col] = data[feature].dt.month
        data[day_col] = data[feature].dt.day
        report['added_features'].extend([year_col, month_col, day_col])

        if day_of_week:
            dow_col = f'{feature}_dayofweek'
            data[dow_col] = data[feature].dt.dayofweek
            report['added_features'].append(dow_col)

        if quarter:
            quarter_col = f'{feature}_quarter'
            data[quarter_col] = data[feature].dt.quarter
            report['added_features'].append(quarter_col)

    # Visualization of data distribution before and after conversion
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[date_features].nunique().to_frame().T,
                    annot=True, cbar=False, cmap=cmap)
        plt.title('Unique Values Before Conversion')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[date_features + report['added_features']
                         ].nunique().to_frame().T, annot=True, cbar=False,
                    cmap=cmap)
        plt.title('Unique Values After Conversion')
        plt.show()

    return (data, report) if return_report else data

def scale_data(
    data: DataFrame, /, 
    method: str = 'norm',
    return_report: bool = False,
    use_sklearn: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Scales numerical columns in the DataFrame using the specified scaling 
    method. 
    
    Optionally returns a report on the scaling process along with 
    visualization.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be scaled.
    method : str
        Scaling method - 'minmax', 'norm', or 'standard'.
    return_report : bool, optional
        If True, returns a report summarizing the scaling process.
    use_sklearn: bool, optional 
        If True and scikit-learn is installed, use its scaling utilities.
    view : bool, optional
        If True, displays a heatmap of the data distribution before and 
        after scaling.
    cmap : str, optional
        The colormap for the heatmap visualization.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        The scaled DataFrame and optionally a report.

    Raises
    ------
    ValueError
        If an invalid scaling method is provided.
        
    Note 
    -----
    Scaling method - 'minmax' or 'standard'.
    'minmax' scales data to the [0, 1] range using the formula:
        
    .. math:: 
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        
    'standard' scales data to zero mean and unit variance using the formula:
        
    .. math:: 
        X_scaled = (X - X.mean()) / X.std()
        
    Example
    -------
    >>> from gofast.tools.baseutils import scale_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> scaled_data, report = scale_data(data, 'minmax',
                                         return_report=True, view=True)
    """
    is_frame (data, df_only=True, raise_exception=True, 
              objname="Exceptionnaly, scaling data")
    numeric_cols = data.select_dtypes(include=['number']).columns
    report = {'method_used': method, 'columns_scaled': list(numeric_cols)}
    
    original_data = data.copy()
    _,method=normalize_string (method, target_strs=(
        'MinMax', "Standard", "Normalization"),return_target_str=True, deep=True)
    # Determine which scaling method to use
    if method not in ['minmax', 'norm', 'standard']:
        raise ValueError("Invalid scaling method. Choose 'minmax',"
                         " 'norm', or 'standard'.")
    if use_sklearn:
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        except ImportError:
            use_sklearn = False

    if not use_sklearn:
        minmax_scale = lambda col: (col - col.min()) / (col.max() - col.min())
        standard_scale = lambda col: (col - col.mean()) / col.std()
        scaling_func = minmax_scale if method == 'minmax' else standard_scale
        data[numeric_cols] = data[numeric_cols].apply(scaling_func)

    # Visualization of data distribution before and after scaling
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data[numeric_cols], annot=True, cbar=False, 
                    cmap=cmap)
        plt.title('Before Scaling')

        plt.subplot(1, 2, 2)
        sns.heatmap(data[numeric_cols], annot=True, cbar=False, cmap=cmap)
        plt.title('After Scaling')
        plt.show()

    return (data, report) if return_report else data

def handle_outliers_in_data(
    data: DataFrame, /, 
    method: str = 'clip', 
    replace_with: str = 'median', 
    lower_quantile: float = 0.01, 
    upper_quantile: float = 0.99,
    return_report: bool = False, 
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (12, 5)
) -> DataFrame:
    """
    Handles outliers in numerical columns of the DataFrame using various 
    methods. 
    
    Optionally, function displays a comparative plot showing the data 
    distribution before and after outlier handling.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame with potential outliers.
    method : str, optional
        Method to handle outliers ('clip', 'remove', 'replace'). 
        Default is 'clip'.
    replace_with : str, optional
        Specifies replacement method ('mean' or 'median') for 'replace'.
        Default is 'median'.
    lower_quantile : float, optional
        Lower quantile for clipping outliers. Default is 0.01.
    upper_quantile : float, optional
        Upper quantile for clipping outliers. Default is 0.99.
    return_report : bool, optional
        If True, returns a report summarizing the outlier handling process.
    view : bool, optional
        If True, displays a comparative plot of the data distribution before 
        and after handling outliers.
    cmap : str, optional
        The colormap for the heatmap visualization. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled and optionally a report dictionary.

    Example
    -------
    >>> from gofast.tools.baseutils import handle_outliers_in_data
    >>> data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    >>> data, report = handle_outliers_in_data(data, method='clip', view=True, 
                                               cmap='plasma', return_report=True)
    """
    is_frame (data, df_only=True, raise_exception=True)
    numeric_cols = data.select_dtypes(include=['number']).columns
    data_before = data.copy()  # Copy of the original data for comparison
    report = {}

    # Handling outliers
    if method == 'clip':
        lower = data[numeric_cols].quantile(lower_quantile)
        upper = data[numeric_cols].quantile(upper_quantile)
        data[numeric_cols] = data[numeric_cols].clip(lower, upper, axis=1)
        report['method'] = 'clip'
    elif method == 'remove':
        # Removing outliers based on quantiles
        lower = data[numeric_cols].quantile(lower_quantile)
        upper = data[numeric_cols].quantile(upper_quantile)
        data = data[(data[numeric_cols] >= lower) & (data[numeric_cols] <= upper)]
        report['method'] = 'remove'
    elif method == 'replace':
        if replace_with not in ['mean', 'median']:
            raise ValueError("Invalid replace_with option. Choose 'mean' or 'median'.")
        replace_func = ( data[numeric_cols].mean if replace_with == 'mean' 
                        else data[numeric_cols].median)
        data[numeric_cols] = data[numeric_cols].apply(lambda col: col.where(
            col.between(col.quantile(lower_quantile), col.quantile(upper_quantile)),
            replace_func(), axis=0))
        report['method'] = 'replace'
    else:
        raise ValueError("Invalid method for handling outliers.")
    
    report['lower_quantile'] = lower_quantile
    report['upper_quantile'] = upper_quantile

    if view:
        # Visualize data distribution before and after handling outliers
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
        
        sns.heatmap(data_before[numeric_cols].isnull(), yticklabels=False, 
                    cbar=False, cmap=cmap, ax=axes[0])
        axes[0].set_title('Before Outlier Handling')

        sns.heatmap(data[numeric_cols].isnull(), yticklabels=False,
                    cbar=False, cmap=cmap, ax=axes[1])
        axes[1].set_title('After Outlier Handling')

        plt.suptitle('Comparative Missing Value Heatmap')
        plt.show()

    return (data, report) if return_report else data

def handle_missing_data(
    data:DataFrame, /, 
    method: Optional[str] = None,  
    fill_value: Optional[Any] = None,
    dropna_threshold: float = 0.5, 
    return_report: bool = False,
    view: bool = False, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> Union[DataFrame, Tuple[DataFrame, dict]]:
    """
    Analyzes patterns of missing data in the DataFrame. 
    
    Optionally, function displays a heatmap before and after handling missing 
    data, and handles missing data based on the specified method.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to analyze and handle missing data.
    method : str, optional
        Method to handle missing data. Options: 'drop_rows', 'drop_cols', 'fill_mean',
        'fill_median', 'fill_value'. If None, no handling is performed.
    fill_value : Any, optional
        Value to use when filling missing data for 'fill_value' method.
    dropna_threshold : float, optional
        Threshold for dropping rows/columns with missing data. 
        Only used with 'drop_rows' or 'drop_cols' method.
    return_report : bool, optional
        If True, returns a tuple of the DataFrame and a report dictionary.
    view : bool, optional
        If True, displays a heatmap of missing data before and after handling.
    cmap : str, optional
        The colormap for the heatmap visualization.
    fig_size : Tuple[int, int], optional
        The size of the figure for the heatmap.
        
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame after handling missing data and optionally a report dictionary.

    Example
    -------
    >>> from gofast.tools.baseutils import handle_missing_data
    >>> data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    >>> updated_data, report = handle_missing_data(
        data, view=True, method='fill_mean', return_report=True)
    """
    is_frame (data, df_only=True, raise_exception=True)
    # Analyze missing data
    original_data = data.copy()
    missing_data = pd.DataFrame(data.isnull().sum(), columns=['missing_count'])
    missing_data['missing_percentage'] = (missing_data['missing_count'] / len(data)) * 100

    # Handling missing data based on method
    handling_methods = {
        'drop_rows': lambda d: d.dropna(thresh=int(dropna_threshold * len(d.columns))),
        'drop_cols': lambda d: d.dropna(axis=1, thresh=int(dropna_threshold * len(d))),
        'fill_mean': lambda d: d.fillna(d.mean()),
        'fill_median': lambda d: d.fillna(d.median()),
        'fill_value': lambda d: d.fillna(fill_value)
    }

    if method in handling_methods:
        if method == 'fill_value' and fill_value is None:
            raise ValueError("fill_value must be specified for 'fill_value' method.")
        data = handling_methods[method](data)
    elif method:
        raise ValueError(f"Invalid method specified: {method}")

    # Visualization of missing data before and after handling
    if view:
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        sns.heatmap(original_data.isnull(), yticklabels=False, cbar=False, 
                    cmap=cmap)
        plt.title('Before Handling Missing Data')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap=cmap)
        plt.title('After Handling Missing Data')
        plt.show()

    # Data report
    data_report = {
        "missing_data_before": original_data.isnull().sum(),
        "missing_data_after": data.isnull().sum(),
        "stats": {
            "method_used": method,
            "fill_value": fill_value if method == 'fill_value' else None,
            "dropna_threshold": dropna_threshold if method in [
                'drop_rows', 'drop_cols'] else None
        },
        "describe": missing_data.describe()
    }
    
    return (data, data_report) if return_report else data

def inspect_data(
    data: DataFrame, /, 
    correlation_threshold: float = 0.8, 
    categorical_threshold: float = 0.75
) -> None:
    """
    Performs an exhaustive inspection of a DataFrame. 
    
    Funtion evaluates data integrity,provides detailed statistics, and offers
    tailored recommendations to ensure data quality for analysis or modeling.

    This function is integral for identifying and understanding various aspects
    of data quality such as missing values, duplicates, outliers, imbalances, 
    and correlations. It offers insights into the data's distribution, 
    variability, and potential issues, guiding users towards effective data 
    cleaning and preprocessing strategies.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be inspected.

    correlation_threshold : float, optional
        The threshold for flagging high correlation between numeric features.
        Features with a correlation above this threshold will be highlighted.
        Default is 0.8.

    categorical_threshold : float, optional
        The threshold for detecting imbalance in categorical variables. If the
        proportion of the most frequent category exceeds this threshold, it will
        be flagged as imbalanced. Default is 0.75.

    Returns
    -------
    None
        Prints a comprehensive report including data integrity assessment, 
        statistics, and recommendations for data preprocessing.

    Example
    -------
    >>> from gofast.tools.baseutils import inspect_data
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    >>>     'A': np.random.normal(0, 1, 100),
    >>>     'B': np.random.normal(5, 2, 100),
    >>>     'C': np.random.randint(0, 100, 100)
    >>> })
    >>> data.iloc[0, 0] = np.nan  # Introduce a missing value
    >>> data.iloc[1] = data.iloc[0]  # Introduce a duplicate row
    >>> inspect_data(data)
    """
    def format_report_section(title, content):
        """
        Formats and prints a section of the report.
        """
        print(f"\033[1m{title}:\033[0m")
        if ( title.lower().find('report')>=0 or title.lower(
           ).find('recomm')>=0): print("-" * (len(title)+1)) 
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {content}")
        print()
        
    def calculate_statistics(d: DataFrame) -> Dict[str, Any]:
        """
        Calculates various statistics for the numerical columns of 
        the DataFrame.
        """
        stats = {}
        numeric_cols = d.select_dtypes(include=[np.number])

        stats['mean'] = numeric_cols.mean()
        stats['std_dev'] = numeric_cols.std()
        stats['percentiles'] = numeric_cols.quantile([0.25, 0.5, 0.75]).T
        stats['min'] = numeric_cols.min()
        stats['max'] = numeric_cols.max()

        return stats
    
    is_frame( data, df_only=True, raise_exception=True,
             objname="Data for inspection")
    is_valid, integrity_report = verify_data_integrity(data)
    stats_report = calculate_statistics(data)
    
    # Display the basic integrity report
    format_report_section("Data Integrity Report", "")
    format_report_section("Missing Values", integrity_report['missing_values'])
    format_report_section("Duplicate Rows", integrity_report['duplicates'])
    format_report_section("Potential Outliers", integrity_report['outliers'])

    # Display the statistics report
    format_report_section("Data Statistics Report", "")
    for stat_name, values in stats_report.items():
        format_report_section(stat_name.capitalize(), values)

    # Recommendations based on the report
    if not is_valid:
        format_report_section("Recommendations", "")
        if integrity_report['missing_values'].any():
            print("- Consider handling missing values using imputation or removal.")
        if integrity_report['duplicates'] > 0:
            print("- Check for and remove duplicate rows to ensure data uniqueness.")
        if any(count > 0 for count in integrity_report['outliers'].values()):
            print("- Investigate potential outliers. Consider removal or transformation.\n")
        
        # Additional checks and recommendations
        # Check for columns with a single unique value
        single_value_columns = [col for col in data.columns if 
                                data[col].nunique() == 1]
        if single_value_columns:
            print("- Columns with a single unique value detected:"
                  f" {single_value_columns}. Consider removing them"
                  " as they do not provide useful information for analysis.")
    
        # Check for data imbalance in categorical variables
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if data[col].value_counts(normalize=True).max() > categorical_threshold:
                print(f"- High imbalance detected in categorical column '{col}'."
                      " Consider techniques to address imbalance, like sampling"
                      " methods or specialized models.")
    
        # Check for skewness in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if abs(data[col].skew()) > 1:
                print(f"- High skewness detected in numeric column '{col}'."
                      " Consider transformations like log, square root, or "
                      "Box-Cox to normalize the distribution.")
        # Normalization for numerical columns
        print("- Evaluate if normalization (scaling between 0 and 1) is "
              "necessary for numerical features, especially for distance-based"
              " algorithms.")
    
        # Correlation check
        correlation_threshold = correlation_threshold  # Arbitrary threshold
        corr_matrix = data[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(
            corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col1, col2) for col1, col2 in zip(
            *np.where(upper_triangle > correlation_threshold))]
    
        if high_corr_pairs:
            print("- Highly correlated features detected:")
            for idx1, idx2 in high_corr_pairs:
                col1, col2 = numeric_cols[idx1], numeric_cols[idx2]
                print(f"  {col1} and {col2} (Correlation > {correlation_threshold})")
        
        # Data type conversions
        print("- Review data types of columns for appropriate conversions"
              " (e.g., converting float to int where applicable).")
                 
def augment_data(
    X: Union[DataFrame, ArrayLike], 
    y: Optional[Union[pd.Series, np.ndarray]] = None, 
    augmentation_factor: int = 2, 
    shuffle: bool = True
) -> Union[Tuple[Union[DataFrame, ArrayLike], Optional[Union[
    Series, ArrayLike]]], Union[DataFrame, ArrayLike]]:
    """
    Augment a dataset by repeating it with random variations to enhance 
    training diversity.

    This function is useful in scenarios with limited data, helping improve the 
    generalization of machine learning models by creating a more diverse training set.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input data, either as a Pandas DataFrame or a NumPy ndarray.

    y : Optional[Union[pd.Series, np.ndarray]], optional
        Target labels, either as a Pandas Series or a NumPy ndarray. If `None`, 
        the function only processes the `X` data. This is useful in unsupervised 
        learning scenarios where target labels may not be applicable.
    augmentation_factor : int, optional
        The multiplier for data augmentation. Defaults to 2 (doubling the data).
    shuffle : bool, optional
        If True, shuffle the data after augmentation. Defaults to True.

    Returns
    -------
    X_augmented : pd.DataFrame or np.ndarray
        Augmented input data in the same format as `X`.
    y_augmented : pd.Series, np.ndarray, or None
        Augmented target labels in the same format as `y`, if `y` is not None.
        Otherwise, None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if the lengths of `X` and `y` 
        are mismatched when `y` is not None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if `X` and `y` have mismatched lengths.

    Examples
    --------
    >>> from gofast.tools.baseutils import augment_data 
    >>> X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
    >>> X_aug, y_aug = augment_data(X, y)
    >>> X_aug.shape, y_aug.shape
    ((4, 2), (4,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> X_aug = augment_data(X, y=None)
    >>> X_aug.shape
    (4, 2)
    """
    from sklearn.utils import shuffle as shuffle_data
    if augmentation_factor < 1:
        raise ValueError("Augmentation factor must be at least 1.")

    is_X_df = isinstance(X, pd.DataFrame)
    is_y_series = isinstance(y, pd.Series) if y is not None else False

    if is_X_df:
        # Separating numerical and categorical columns
        num_columns = X.select_dtypes(include=['number']).columns
        cat_columns = X.select_dtypes(exclude=['number']).columns

        # Augment only numerical columns
        X_num = X[num_columns]
        X_num_augmented = np.concatenate([X_num] * augmentation_factor)
        X_num_augmented += np.random.normal(loc=0.0, scale=0.1 * X_num.std(axis=0), 
                                            size=X_num_augmented.shape)
        # Repeat categorical columns without augmentation
        X_cat_augmented = pd.concat([X[cat_columns]] * augmentation_factor
                                    ).reset_index(drop=True)

        # Combine numerical and categorical data
        X_augmented = pd.concat([pd.DataFrame(
            X_num_augmented, columns=num_columns), X_cat_augmented], axis=1)
   
    else:
        # If X is not a DataFrame, it's treated as a numerical array
        X_np = np.asarray(X, dtype= float) 
        X_augmented = np.concatenate([X_np] * augmentation_factor)
        X_augmented += np.random.normal(
            loc=0.0, scale=0.1 * X_np.std(axis=0), size=X_augmented.shape)

    y_np = np.asarray(y) if y is not None else None
    
    # Shuffle if required
    if y_np is not None:
        check_consistent_length(X, y_np )
        y_augmented = np.concatenate([y_np] * augmentation_factor)
        if shuffle:
            X_augmented, y_augmented = shuffle_data(X_augmented, y_augmented)
        if is_y_series:
            y_augmented = pd.Series(y_augmented, name=y.name)
            
        return X_augmented, y_augmented

    else:
        if shuffle:
            X_augmented = shuffle_data(X_augmented)
            
        return X_augmented

def _is_categorical(y: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Determine if the target variable(s) is categorical.

    Parameters
    ----------
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s).

    Returns
    -------
    bool
        True if the target variable(s) is categorical, False otherwise.
    """
    if isinstance(y, pd.DataFrame):
        return all(y.dtypes.apply(lambda dtype: dtype.kind in 'iOc'))
    return y.dtype.kind in 'iOc'

def _encode_target(y: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Encode the target variable(s) if it is categorical.

    Parameters
    ----------
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s).

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Encoded target variable(s).
    """
    from sklearn.preprocessing import LabelEncoder
    
    if isinstance(y, pd.DataFrame):
        encoder = LabelEncoder()
        return y.apply(lambda col: encoder.fit_transform(col))
    elif y.dtype.kind == 'O':
        encoder = LabelEncoder()
        return pd.Series(encoder.fit_transform(y), index=y.index)
    return y

def _prepare_data(
        data: pd.DataFrame, 
        target_column: Union[str, List[str], pd.Series, np.ndarray]
        ) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Prepare the feature matrix X and target vector y from the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing features and a target variable.
    target_column : Union[str, List[str], pd.Series, np.ndarray]
        The target variable's identifier(s) or the target variable itself.

    Returns
    -------
    X : pd.DataFrame
        The feature matrix after removing the target column(s) if specified by name.
    y : Union[pd.Series, pd.DataFrame]
        The target vector or DataFrame for multiple labels.

    Raises
    ------
    ValueError
        If the target_column length does not match the number of samples in data.
    KeyError
        If the target_column specified by name is not found in data.
    """
    if isinstance(target_column, (str, list)):
        target_column= list(is_iterable(
            target_column, exclude_string= True, transform =True )
            )
        exist_features(data, features= target_column )
        y = data[target_column]
        X = data.drop(target_column, axis=1)
    elif isinstance(target_column, (pd.Series, np.ndarray)):
        # if len(target_column) != len(data):
        #     raise ValueError("Length of the target array/Series must match the number of samples in 'data'.")
        y = pd.Series(target_column, index=data.index) if isinstance(target_column, np.ndarray) else target_column
        X = data
    else:
        raise ValueError("Invalid type for 'target_column'. Must be str, list, pd.Series, or np.ndarray.")
    
    check_consistent_length(X, y)
    
    return X, y

def assess_outlier_impact(
    data: pd.DataFrame, /, 
    target_column: Union[str, List[str], pd.Series, np.ndarray],
    test_size: float = 0.2, 
    random_state: int = 42, 
    verbose: bool = False) -> Tuple[float, float]:
    """
    Assess the impact of outliers on the predictive performance of a model. 
    
    Applicable for both regression and classification tasks, including 
    multi-label targets.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing features and a target variable.
        
    target_column : Union[str, List[str], pd.Series, np.ndarray]
        The name of the target variable column(s) in the DataFrame, or the target 
        variable array/Series itself. If a string or list of strings is provided, 
        the column(s) will be used as the target variable and removed from the data.
        
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
        
    random_state : int, optional (default=42)
        The random state to use for reproducible train-test splits.
    
    verbose : bool
        If True, prints the evaluation metric with and without outliers, 
        and the impact message.
        
    Returns
    -------
    Tuple[float, float]
        A tuple containing the evaluation metric (MSE for regression or 
                                                  accuracy for classification)
        of the model's predictions on the test set with outliers present and 
        with outliers removed.
        
     Raises:
     -------
     KeyError
         If the target column is not present in the DataFrame.
         
     ValueError
         If the test size is not between 0 and 1.
         
    Examples:
    ---------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import assess_outlier_impact
    >>> df = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })
    >>> mse_with_outliers, mse_without_outliers = assess_outlier_impact(df, 'target')
    >>> print('MSE with outliers:', mse_with_outliers)
    >>> print('MSE without outliers:', mse_without_outliers)

    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    is_frame (data, df_only= True, raise_exception= True )
    X, y = _prepare_data(data, target_column)
    # Determine if the task is regression or classification
    is_categorical = _is_categorical(y)
    if is_categorical:
        model = LogisticRegression()
        metric = accuracy_score
        metric_name = "Accuracy"
        y = _encode_target(y)
    else:
        model = LinearRegression()
        metric = mean_squared_error
        metric_name = "MSE"
    # Ensure y is a np.ndarray for model fitting
    y = y if isinstance(y, np.ndarray) else y.values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Define an internal function for model evaluation to keep the main logic clean
    def _evaluate_model(X_train: pd.DataFrame, y_train: np.ndarray,
                        X_test: pd.DataFrame, y_test: np.ndarray, 
                        model, metric) -> float:
        """ Train the model and evaluate its performance on the test set."""
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return metric(y_test, predictions)

    # Evaluate the model on the original data
    original_metric = _evaluate_model(X_train, y_train, X_test, y_test, model, metric)

    # Identify and remove outliers using the IQR method
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1
    is_not_outlier = ~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)
    X_train_filtered = X_train[is_not_outlier]
    y_train_filtered = y_train[is_not_outlier]

    # Evaluate the model on the filtered data
    filtered_metric = _evaluate_model(X_train_filtered, y_train_filtered, X_test, y_test, model, metric)

    # Print results if verbose is True
    if verbose:
        print(f'{metric_name} with outliers in the training set: {original_metric}')
        print(f'{metric_name} without outliers in the training set: {filtered_metric}')
        # Check the impact
        if not is_categorical and filtered_metric < original_metric or \
           is_categorical and filtered_metric > original_metric:
            print('Outliers appear to have a negative impact on the model performance.')
        else:
            print('Outliers do not appear to have a significant negative'
                  ' impact on the model performance.')

    return original_metric, filtered_metric

def transform_dates(
    data: DataFrame, /, 
    transform: bool = True, 
    fmt: Optional[str] = None, 
    return_dt_columns: bool = False,
    include_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    force: bool = False,
    errors: str = 'coerce',
    **dt_kws
) -> Union[pd.DataFrame, List[str]]:
    """
    Detects and optionally transforms columns in a DataFrame that can be 
    interpreted as dates. 
    
    Funtion uses advanced parameters for greater control over the 
    conversion process.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to inspect and process.
    transform : bool, optional
        Determines whether to perform the conversion of detected datetime 
        columns. Defaults to True.
    fmt : str or None, optional
        Specifies the datetime format string to use for conversion. If None, 
        pandas will infer the format.
    return_dt_columns : bool, optional
        If True, the function returns a list of column names detected 
        (and potentially converted) as datetime. 
        Otherwise, it returns the modified DataFrame or the original DataFrame
        if no transformation is performed.
    include_columns : List[str] or None, optional
        Specifies a list of column names to consider for datetime conversion. 
        If None, all columns are considered.
    exclude_columns : List[str] or None, optional
        Specifies a list of column names to exclude from datetime conversion. 
        This parameter is ignored if `include_columns` is provided.
    force : bool, optional
        If True, forces the conversion of columns to datetime objects, even 
        for columns with mixed or unexpected data types.
    errors : str, optional
        Determines how to handle conversion errors. Options are 'raise', 
        'coerce', and 'ignore' (default is 'coerce').
        'raise' will raise an exception for any errors, 'coerce' will convert 
        problematic data to NaT, and 'ignore' will
        return the original data without conversion.
    **dt_kws : dict
        Additional keyword arguments to be passed to `pd.to_datetime`.

    Returns
    -------
    Union[pd.DataFrame, List[str]]
        Depending on `return_dt_columns`, returns either a list of column names 
        detected as datetime or the DataFrame with the datetime conversions 
        applied.

    Examples
    --------
    >>> from gofast.tools.baseutils import transform_dates
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> transform_dates(data, fmt='%Y-%m-%d', return_dt_columns=True)
    ['date', 'timestamp']

    >>> transform_dates(data, include_columns=['date', 'timestamp'], 
    ...                    errors='ignore').dtypes
    date          datetime64[ns]
    value                  int64
    timestamp     datetime64[ns]
    text                  object
    dtype: object
    """
    # Use the helper function to identify potential datetime columns
    potential_dt_columns = detect_datetime_columns(data)
    
    # Filter columns based on include/exclude lists if provided
    if include_columns is not None:
        datetime_columns = [col for col in include_columns 
                            if col in potential_dt_columns]
    elif exclude_columns is not None:
        datetime_columns = [col for col in potential_dt_columns 
                            if col not in exclude_columns]
    else:
        datetime_columns = potential_dt_columns

    df = data.copy()
    
    if transform:
        for col in datetime_columns:
            if force or col in datetime_columns:
                df[col] = pd.to_datetime(df[col], format=fmt,
                                         errors=errors, **dt_kws)
    
    if return_dt_columns:
        return datetime_columns
    
    return df
    
def detect_datetime_columns(data: DataFrame, / ) -> List[str]:
    """
    Detects columns in a DataFrame that can be interpreted as date and time,
    with an improved check to avoid false positives on purely numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    List[str]
        A list of column names that can potentially be formatted as datetime objects.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> detect_datetime_columns(data)
    ['date', 'timestamp']
    """
    datetime_columns = []

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].dropna().empty:
            # Skip numeric columns with no values, as they cannot be dates
            continue
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                # Attempt conversion on columns with string-like or mixed types
                _ = pd.to_datetime(data[col], errors='raise')
                datetime_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, skip the column.
                continue

    return datetime_columns

def merge_frames_on_index(
    *data: DataFrame, 
    index_col: str, 
    join_type: str = 'outer', 
    axis: int = 1, 
    ignore_index: bool = False, 
    sort: bool = False
    ) -> DataFrame:
    """
    Merges multiple DataFrames based on a specified column set as the index.

    Parameters
    ----------
    *data : pd.DataFrame
        Variable number of pandas DataFrames to merge.
    index_col : str
        The name of the column to set as the index in each DataFrame before merging.
    join_type : str, optional
        The type of join to perform. One of 'outer', 'inner', 'left', or 'right'.
        Defaults to 'outer'.
    axis : int, optional
        The axis to concatenate along. {0/'index', 1/'columns'}, default 1/'columns'.
    ignore_index : bool, optional
        If True, the resulting axis will be labeled 0, 1, …, n - 1. Default False.
    sort : bool, optional
        Sort non-concatenation axis if it is not already aligned. Default False.

    Returns
    -------
    pd.DataFrame
        A single DataFrame resulting from merging the input DataFrames based 
        on the specified index.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import merge_frames_on_index
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'Key': ['K0', 'K1', 'K2']})
    >>> df2 = pd.DataFrame({'B': [4, 5, 6], 'Key': ['K0', 'K1', 'K2']})
    >>> merged_df = merge_frames_on_index(df1, df2, index_col='Key')
    >>> print(merged_df)

    Note: This function sets the specified column as the index for each 
    DataFrame if it is not already.
    If the column specified is not present in any of the DataFrames,
    a KeyError will be raised.
    """
    # Ensure all provided data are pandas DataFrames
    if not all(isinstance(df, pd.DataFrame) for df in data):
        raise TypeError("All data provided must be pandas DataFrames.")
        
    # Check if the specified index column exists in all DataFrames
    for df in data:
        if index_col not in df.columns:
            raise KeyError(f"The column '{index_col}' was not found in DataFrame.")

    # Set the specified column as the index for each DataFrame
    indexed_dfs = [df.set_index(index_col) for df in data]

    # Concatenate the DataFrames based on the provided parameters
    merged_df = pd.concat(indexed_dfs, axis=axis, join=join_type,
                          ignore_index=ignore_index, sort=sort)

    return merged_df

def apply_tfidf_vectorization(
    data: DataFrame,/,
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization 
    to one or more text columns in a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    TF-IDF method weighs the words based on their occurrence in a document 
    relative to their frequency across all documents, helping to highlight 
    words that are more interesting, i.e., frequent in a document but not 
    across documents.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the TF-IDF vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the TF-IDF features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import apply_tfidf_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_tfidf_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(text_columns, exclude_string= True, transform=True)
    tfidf_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        column_tfidf_df = _generate_tfidf_features(
            handled_data, max_features, stop_words
            )
        tfidf_features_df = pd.concat(
            [tfidf_features_df, column_tfidf_df], 
            axis=1
            )

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat(
        [data.reset_index(drop=True), tfidf_features_df.reset_index(drop=True)
         ], axis=1
    )

    return prepared_data

def _handle_missing_values(
        column_data: pd.Series, missing_value_handling: str, fill_value: str = ''
   ) -> pd.Series:
    """
    Handles missing values in a pandas Series according to the specified method.

    Parameters
    ----------
    column_data : pd.Series
        The Series (column) from the DataFrame for which missing values 
        need to be handled.
    missing_value_handling : str
        The method for handling missing values: 'fill', 'drop', or 'ignore'.
    fill_value : str, optional
        The value to use for filling missing values if `missing_value_handling`
        is 'fill'. Defaults to an empty string.

    Returns
    -------
    pd.Series
        The Series with missing values handled according to the specified method.

    Raises
    ------
    ValueError
        If an invalid `missing_value_handling` option is provided.
    """
    if missing_value_handling == 'fill':
        return column_data.fillna(fill_value)
    elif missing_value_handling == 'drop':
        return column_data.dropna()
    elif missing_value_handling == 'ignore':
        return column_data
    else:
        raise ValueError("Invalid missing_value_handling option. Choose"
                         " 'fill', 'drop', or 'ignore'.")
def _generate_tfidf_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]) -> pd.DataFrame:
    """
    Generates TF-IDF features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the TF-IDF vectorizer. Can be 'english' 
        or a custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the TF-IDF features for the text data.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    return pd.DataFrame(tfidf_features, columns=feature_names, index=text_data.index)

def apply_bow_vectorization(
    data: pd.DataFrame, /, 
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
 ) -> pd.DataFrame:
    """
    Applies Bag of Words (BoW) vectorization to one or more text columns in 
    a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    Bow is a simpler approach that creates a vocabulary of all the unique 
    words in the dataset and then models each text as a count of the number 
    of times each word appears.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the BoW vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the BoW features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import apply_bow_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_bow_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)

    bow_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value)
        column_bow_df = _generate_bow_features(
            handled_data, max_features, stop_words)
        bow_features_df = pd.concat([bow_features_df, column_bow_df], axis=1)

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True),
                               bow_features_df.reset_index(drop=True)],
                              axis=1
                              )

    return prepared_data

def apply_word_embeddings(
    data: DataFrame,/, 
    text_columns: Union[str, List[str]],
    embedding_file_path: str,
    n_components: int = 50,
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies word embedding vectorization followed by dimensionality reduction 
    to text columns in a pandas DataFrame.
    
    This process converts text data into a numerical form that captures 
    semantic relationships between words, making it suitable for use in machine
    learning models. The function leverages pre-trained word embeddings 
    (e.g., Word2Vec, GloVe) to represent words in a high-dimensional space and 
    then applies PCA (Principal Component Analysis) to reduce the
    dimensionality of these embeddings to a specified number of components.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to be processed.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data to be 
        vectorized. Can be a single column name or a list of names for multiple
        columns.
    embedding_file_path : str
        The file path to the pre-trained word embeddings. This file should be 
        in a format compatible with Gensim's KeyedVectors, such as Word2Vec's .
        bin format or GloVe's .txt format.
    n_components : int, optional
        The number of dimensions to reduce the word embeddings to using PCA. 
        Defaults to 50, balancing between retaining
        semantic information and ensuring manageability for machine learning models.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. Options are:
        - 'fill': Replace missing values with `fill_value`.
        - 'drop': Remove rows with missing values in any of the specified text columns.
        - 'ignore': Leave missing values as is, which may affect the embedding process.
        Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` 
        if `missing_value_handling` is 'fill'. This can be an empty string 
        (default) or any placeholder text.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True, removing the text
        columns to only include the generated features and original non-text data.

    Returns
    -------
    pd.DataFrame
        A DataFrame consisting of the original DataFrame 
        (minus the text columns if `drop_text_columns` is True) concatenated
        with the dimensionality-reduced word embedding features. The new 
        features are numerical and ready for use in machine learning models.

    Examples
    --------
    Assuming we have a DataFrame `df` with a text column 'reviews' and 
    pre-trained word embeddings stored at 'path/to/embeddings.bin':

    >>> import pandas as pd 
    >>> from gofast.tools.baseutils import apply_word_embeddings
    >>> df = pd.DataFrame({'reviews': [
    ...  'This product is great', 'Terrible customer service', 'Will buy again', 
    ... 'Not worth the price']})
    >>> processed_df = apply_word_embeddings(df,
    ...                                      text_columns='reviews',
    ...                                      embedding_file_path='path/to/embeddings.bin',
    ...                                      n_components=50,
    ...                                      missing_value_handling='fill',
    ...                                      fill_value='[UNK]',
    ...                                      drop_text_columns=True)
    >>> processed_df.head()

    This will create a DataFrame with 50 new columns, each representing a 
    component of the reduced dimensionality word embeddings, ready for further 
    analysis or machine learning.
    """
    embeddings = _load_word_embeddings(embedding_file_path)
    
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    all_reduced_embeddings = []

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        avg_embeddings = _average_word_embeddings(handled_data, embeddings)
        reduced_embeddings = _reduce_dimensions(avg_embeddings, n_components)
        all_reduced_embeddings.append(reduced_embeddings)

    # Combine reduced embeddings into a DataFrame
    embeddings_df = pd.DataFrame(np.hstack(all_reduced_embeddings))

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True), 
                               embeddings_df.reset_index(drop=True)],
                              axis=1
                              )

    return prepared_data

def _generate_bow_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]
    ) -> pd.DataFrame:
    """
    Generates Bag of Words (BoW) features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the BoW vectorizer. Can be 'english' or a 
        custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the BoW features for the text data.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    bow_vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    bow_features = bow_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'bow_{i}' for i in range(bow_features.shape[1])]
    return pd.DataFrame(bow_features, columns=feature_names, index=text_data.index)

def _load_word_embeddings(embedding_file_path: str):
    """
    Loads pre-trained word embeddings from a file.

    Parameters
    ----------
    embedding_file_path : str
        Path to the file containing the pre-trained word embeddings.

    Returns
    -------
    KeyedVectors
        The loaded word embeddings.
    """
    import_optional_dependency("gensim", extra=(
        "Word-Embeddings expect 'gensim'to be installed.")
        )
    from gensim.models import KeyedVectors
    embeddings = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)
    
    return embeddings

def _average_word_embeddings(
        text_data: Series, embeddings
    ) -> ArrayLike:
    """
    Generates an average word embedding for each text sample in a Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    embeddings : KeyedVectors
        The pre-trained word embeddings.

    Returns
    -------
    np.ndarray
        An array of averaged word embeddings for the text data.
    """
    def get_embedding(word):
        try:
            return embeddings[word]
        except KeyError:
            return np.zeros(embeddings.vector_size)

    avg_embeddings = text_data.apply(lambda x: np.mean(
        [get_embedding(word) for word in x.split() if word in embeddings],
        axis=0)
        )
    return np.vstack(avg_embeddings)

def _reduce_dimensions(
        embeddings: ArrayLike, n_components: int = 50
        ) -> ArrayLike:
    """
    Reduces the dimensionality of word embeddings using PCA.

    Parameters
    ----------
    embeddings : np.ndarray
        The word embeddings array.
    n_components : int
        The number of dimensions to reduce to.

    Returns
    -------
    np.ndarray
        The dimensionality-reduced word embeddings.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def boxcox_transformation(
    data: DataFrame, 
    columns: Optional[Union[str, List[str]]] = None, 
    min_value: float = 1, 
    adjust_non_positive: str = 'skip',
    verbose: int = 0, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> (DataFrame, Dict[str, Optional[float]]):
    """
    Apply Box-Cox transformation to each numeric column of a pandas DataFrame.
    
    The Box-Cox transformation can only be applied to positive data. This function
    offers the option to adjust columns with non-positive values by either 
    skipping those columns or adding a constant to make all values positive 
    before applying the transformation.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing numeric data to transform. Non-numeric columns 
        will be ignored.
    columns : str or list of str, optional
        List of column names to apply the Box-Cox transformation to. If None, 
        the transformation is applied to all numeric columns in the DataFrame.
    min_value : float or int, optional
        The minimum value to be considered positive. Default is 1. Values in 
        columns must be greater than this minimum to apply the Box-Cox 
        transformation.
    adjust_non_positive : {'skip', 'adjust'}, optional
        Determines how to handle columns with values <= min_value:
            - 'skip': Skip the transformation for these columns.
            - 'adjust': Add a constant to all elements in these columns to 
              make them > min_value before applying the transformation.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = print messages about columns being 
        skipped or adjusted.
        
    view : bool, optional
        If True, displays visualizations of the data before and after 
        transformation.
    cmap : str, optional
        The colormap for visualizing the data distributions. Default is 
        'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualizations. Default is (12, 5).

    Returns
    -------
    transformed_data : pandas.DataFrame
        The DataFrame after applying the Box-Cox transformation to eligible 
        columns.
    lambda_values : dict
        A dictionary mapping column names to the lambda value used for the 
        Box-Cox transformation. Columns that were skipped or not numeric will 
        have a lambda value of None.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.tools.baseutils import boxcox_transformation
    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({
    ...     'A': np.random.rand(10) * 100,
    ...     'B': np.random.normal(loc=50, scale=10, size=10),
    ...     'C': np.random.randint(1, 10, size=10)  # Ensure positive values for example
    ... })
    >>> transformed_data, lambda_values = boxcox_transformation(
    ...     data, columns=['A', 'B'], adjust_non_positive='adjust', verbose=1)
    >>> print(transformed_data.head())
    >>> print(lambda_values)
    """
    is_frame (data, df_only=True, raise_exception= True )
    transformed_data = pd.DataFrame()
    lambda_values = {}

    columns = is_iterable(columns, exclude_string=True, transform= True )
    if columns is not None:
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols and verbose:
            print(f"Warning: Columns {missing_cols} not found in DataFrame."
                  " Skipping these columns.")
        columns = [col for col in columns if col in data.columns]

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_transform = columns if columns is not None else numeric_columns
    
    _, adjust_non_positive= normalize_string(adjust_non_positive, target_strs=(
        "adjust", "skip"), return_target_str= True, raise_exception=True, 
        error_msg= ("`adjust_non_positive` argument expects ['skip', 'adjust']"
                    f" Got: {adjust_non_positive!r}")
        )
    for column in columns_to_transform:
        if column in numeric_columns:
            col_data = data[column]
            if adjust_non_positive == 'adjust' and (col_data <= min_value).any():
                adjustment = min_value - col_data.min() + 1
                col_data += adjustment
                transformed, fitted_lambda = stats.boxcox(col_data)
            elif (col_data > min_value).all():
                transformed, fitted_lambda = stats.boxcox(col_data)
            else:
                transformed = col_data.copy()
                fitted_lambda = None
                if verbose:
                    print(f"Column '{column}' skipped: contains values <= {min_value}.")
            transformed_data[column] = transformed
            lambda_values[column] = fitted_lambda
        else:
            if verbose:
                print(f"Column '{column}' is not numeric and will be skipped.")

    # Include non-transformed columns in the returned DataFrame
    for column in data.columns:
        if column not in transformed_data:
            transformed_data[column] = data[column]
            
    if view:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0]*2, fig_size[1]))
        
        # Determine columns for heatmap
        heatmap_columns = columns if columns is not None else data.select_dtypes(
            include=[np.number]).columns
        heatmap_columns = [col for col in heatmap_columns if col in data.columns 
                           and np.issubdtype(data[col].dtype, np.number)]
        # Original data heatmap
        sns.heatmap(data[heatmap_columns].corr(), ax=axs[0], annot=True, cmap=cmap)
        axs[0].set_title('Correlation Matrix Before Transformation')
        
        # Apply Box-Cox Transformation
        # Transformed data heatmap
        sns.heatmap(transformed_data[heatmap_columns].corr(), ax=axs[1], 
                    annot=True, cmap=cmap)
        axs[1].set_title('Correlation Matrix After Transformation')
        
        plt.tight_layout()
        plt.show()
    
    return transformed_data, lambda_values

def check_missing_data(
    data: DataFrame, /, 
    view: bool = False,
    explode: Optional[Union[Tuple[float, ...], str]] = None,
    shadow: bool = True,
    startangle: int = 90,
    cmap: str = 'viridis',
    autopct: str = '%1.1f%%',
    verbose: int = 0
) -> DataFrame:
    """
    Check for missing data in a DataFrame and optionally visualize the 
    distribution of missing data with a pie chart.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to check for missing data.
    view : bool, optional
        If True, displays a pie chart visualization of the missing data 
        distribution.
    explode : tuple of float, or 'auto', optional
        - If a tuple, it should have a length matching the number of columns 
          with missing data, indicating how far from the center the slice 
          for that column will be.
        - If 'auto', the slice with the highest percentage of missing data will 
          be exploded. If the length does not match, an error is raised.
    shadow : bool, optional
        If True, draws a shadow beneath the pie chart.
    startangle : int, optional
        The starting angle of the pie chart. If greater than 360 degrees, 
        the value is adjusted using modulo operation.
    cmap : str, optional
        The colormap to use for the pie chart.
    autopct : str, optional
        String format for the percentage of each slice in the pie chart. 
    verbose : int, optional
        If set, prints messages about automatic adjustments.

    Returns
    -------
    missing_stats : pandas.DataFrame
        A DataFrame containing the count and percentage of missing data in 
        each column that has missing data.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.baseutils import check_missing_data
    >>> # Create a sample DataFrame with missing values
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': [None, 2, 3, 4],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> missing_stats = check_missing_data(data, view=True, explode='auto',
                                           shadow=True, startangle=270, verbose=1)
    """
    def validate_autopct_format(autopct: str) -> bool:
        """
        Validates the autopct format string for matplotlib pie chart.

        Parameters
        ----------
        autopct : str
            The format string to validate.

        Returns
        -------
        bool
            True if the format string is valid, False otherwise.
        """
        # A regex pattern that matches strings like '%1.1f%%', '%1.2f%%', etc.
        # This pattern checks for the start of the string (%), optional flags,
        # optional width, a period, precision, the 'f' specifier, and ends with '%%'
        pattern = r'^%[0-9]*\.?[0-9]+f%%$'
        return bool(re.match(pattern, autopct))
    
    is_frame( data, df_only= True, raise_exception= True )
    missing_count = data.isnull().sum()
    missing_count = missing_count[missing_count > 0]
    missing_percentage = (missing_count / len(data)) * 100
    missing_stats = pd.DataFrame({'Count': missing_count,
                                  'Percentage': missing_percentage})
    
    if view and not missing_count.empty:
        labels = missing_stats.index.tolist()
        sizes = missing_stats['Percentage'].values.tolist()
        if explode == 'auto':
            # Dynamically create explode data 
            explode = [0.1 if i == sizes.index(max(sizes)) else 0 
                       for i in range(len(sizes))]
        elif explode is not None:
            if len(explode) != len(sizes):
                raise ValueError(
                    f"The length of 'explode' ({len(explode)}) does not match "
                    f"the number of columns with missing data ({len(sizes)})."
                    " Set 'explode' to 'auto' to avoid this error.")
        
        if startangle > 360:
            startangle %= 360
            if verbose:
                print("Start angle greater than 180 degrees. Using modulo "
                      f"to adjust: startangle={startangle}")
        
        if not validate_autopct_format(autopct):
            raise ValueError("`autopct` format is not valid. It should be a"
                             "  format string like '%1.1f%%'.")
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct=autopct,
               shadow=shadow, startangle=startangle, colors=plt.get_cmap(cmap)(
                   np.linspace(0, 1, len(labels))))
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        ax.set_title('Missing Data Distribution')
        plt.show()

    return missing_stats

