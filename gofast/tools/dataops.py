# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
from __future__ import annotations, print_function 
import os
import re
import h5py
import shutil 
import pathlib
import datetime
import warnings 
from scipy import stats
from six.moves import urllib 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm

from ..api.formatter import MultiFrameFormatter, format_iterable 
from ..api.property import  Config
from ..api.summary import ReportFactory, Summary
from ..api.types import Any,  List,  DataFrame, Optional, Series, Array1D 
from ..api.types import Dict, Union, TypeGuard, Tuple, ArrayLike, Callable
from ..api.types import BeautifulSoupTag
from ..decorators import Deprecated, isdf, Dataify, DynamicMethod
from ..decorators import DataTransformer 
from ..exceptions import FileHandlingError 
from .baseutils import save_or_load 
from .coreutils import is_iterable, ellipsis2false,smart_format, validate_url 
from .coreutils import to_numeric_dtypes, assert_ratio, exist_features
from .coreutils import normalize_string
from .funcutils import ensure_pkg
from .validator import  build_data_if, is_frame, parameter_validator  
from .validator import check_consistent_length

def summarize_text_columns(
    data: DataFrame, /, 
    text_columns: List[str], 
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
    >>> from gofast.tools.dataops import summarize_text_columns
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
    
    for column_name in text_columns:
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
        data.drop(columns=text_columns, inplace=True)
        data.columns = [ c.replace ("_encoded", '') for c in data.columns]

    return data

def simple_extractive_summary(
    texts: List[str], raise_exception: bool = True,
    encode: bool = False
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
    >>> from gofast.tools.dataops import simple_extractive_summary
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

@isdf 
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
    >>> from gofast.tools.dataops import format_long_column_names
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
    >>> from gofast.tools.dataops import enrich_data_spectrum
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

@isdf 
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
        Default is ``iqr``. 
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
    >>> from gofast.tools.dataops import sanitize
    >>> data = {'A': [1, 2, None, 4], 'B': ['X', 'Y', 'Y', None], 'C': [1, 1, 2, 2]}
    >>> df = pd.DataFrame(data)
    >>> cleaned_df = sanitize(df, fill_missing='median', remove_duplicates=True,
                                outlier_method='z_score', 
                                consistency_transform='lower', threshold=3)
    >>> print(cleaned_df)
    """
    # validate input parameters 
    outlier_method = parameter_validator(
        'outlier_method', ['z_score', 'iqr'])(outlier_method)
    fill_missing = parameter_validator(
        'fill_missing', ['median', 'mean', 'mode'])(fill_missing)
    consistency_transform = parameter_validator(
        'consistency_transform', ['lower', 'upper'])(consistency_transform)
 
    data = build_data_if(data, to_frame=True, force=True, input_name="feature_", 
                         raise_warning='mute')
    data = to_numeric_dtypes( data ) # verify integrity 
    df_cleaned = data.copy()
    if fill_missing:
        fill_methods = {
            'median': data.median(numeric_only=True),
            'mean': data.mean(numeric_only=True),
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
    gofast.tools.dataops.save_or_load: 
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
    

@ensure_pkg("requests")
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
    >>> from gofast.tools.dataops import request_data
    >>> response = request_data('https://api.github.com/user',
                                auth=('user', 'pass'), as_json=True)
    >>> print(response)
    """

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

@Deprecated("Deprecated function. Should be remove next release."
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
    >>> from gofast.tools.dataops import get_remote_data
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



@ensure_pkg("bs4", " Needs `BeautifulSoup` from `bs4` package" )
@ensure_pkg("requests")
def scrape_web_data(
    url: str, 
    element: str, 
    class_name: Optional[str] = None, 
    attributes: Optional[dict] = None, 
    parser: str = 'html.parser'
    ) -> List[BeautifulSoupTag[str]]:
    """
    Scrape data from a web page using BeautifulSoup.

    Parameters
    ----------
    url : str
        The URL of the web page to scrape.
    element : str
        The HTML element to search for.
    class_name : str, optional
        The class attribute of the HTML element to narrow down the search.
        Default is None.
    attributes : dict, optional
        Additional attributes of the HTML element to narrow down the search. 
        Default is None.
    parser : str, optional
        The parser used by BeautifulSoup. Default is 'html.parser'.

    Returns
    -------
    list of bs4.element.Tag
        A list of BeautifulSoup Tag objects that match the search query.

    Examples
    --------
    >>> from gofast.tools.dataops import scrape_web_data
    >>> url = 'https://example.com'
    >>> element = 'div'
    >>> class_name = 'content'
    >>> data = scrape_web_data(url, element, class_name)
    >>> for item in data:
    ...     print(item.text)

    >>> url = 'https://example.com/articles'
    >>> element = 'h1'
    >>> data = scrape_web_data(url, element)
    >>> for header in data:
    ...    print(header.text)  # prints the text of each <h1> tag

    >>> url = 'https://example.com/products'
    >>> element = 'section'
    >>> attributes = {'id': 'featured-products'}
    >>> data = scrape_web_data(url, element, attributes=attributes)
    >>> # prints the text of each section with id 'featured-products'
    >>> for product in data:
    ...     print(product.text)  
    """

    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, parser)
        if class_name:
            elements = soup.find_all(element, class_=class_name)
        elif attributes:
            elements = soup.find_all(element, **attributes)
        else:
            elements = soup.find_all(element)
        return elements
    else:
        response.raise_for_status()


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
    >>> from gofast.tools.dataops import handle_datasets_with_hdfstore
    
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
    Tuple[bool, Report]
        A tuple containing:
        - A boolean indicating if the data passed all integrity checks 
          (True if no issues are found).
        - A dictionary with the details of the checks, including counts of 
        missing values, duplicates, and outliers by column.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import verify_data_integrity
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
    report['integrity_checks']='Passed' if is_valid else 'Failed'
    # make a report obj 
    report_obj= ReportFactory(title ="Data Integrity", **report )
    report_obj.add_mixed_types(report, table_width= 90)
    
    return is_valid, report_obj

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
    >>> from gofast.tools.dataops import audit_data
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
        update_report(handle_outliers_in(
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
    
    # make a report obj 
    if return_report: 
        report_obj= ReportFactory(title ="Data Audit")
        report_obj.add_mixed_types(report, table_width= 90)
    
    return (data, report_obj) if return_report else data

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
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import handle_categorical_features
    >>> data = pd.DataFrame({'A': [1, 2, 1, 3], 'B': range(4)})
    >>> updated_data, report = handle_categorical_features(
        data, categorical_threshold=3, return_report=True, view=True)
    >>> report.converted_columns
    ['A']
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
    # make a report obj 
    report_obj= ReportFactory(title ="Categorical Features Handling", **report )
    report_obj.add_mixed_types(report, table_width= 90,  )
    
    return (data, report_obj) if return_report else data

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
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import convert_date_features
    >>> data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02']})
    >>> updated_data, report = convert_date_features(
        data, ['date'], day_of_week=True, quarter=True, return_report=True, view=True)
    >>> report.converted_columns
    ['date']
    >>> report.added_features
    ['date_year', 'date_month', 'date_day', 'date_dayofweek', 'date_quarter']
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

    report_obj= ReportFactory(title ="Date Features Conversion", **report )
    report_obj.add_mixed_types(report, table_width= 90,)
    return (data, report_obj) if return_report else data

@isdf 
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
    >>> from gofast.tools.dataops import scale_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> scaled_data, report = scale_data(data, 'minmax', return_report=True, view=True)
    >>> print(report) 
    >>> report.method_used
    'minmax'
    >>> report.columns_scaled
    ['A', 'B']
    
    """
    is_frame (data, df_only=True, raise_exception=True, 
              objname="Exceptionnaly, scaling data")
    numeric_cols = data.select_dtypes(include=['number']).columns
    report = {'method_used': method, 'columns_scaled': list(numeric_cols)}
    
    original_data = data.copy()
    method=normalize_string (method, target_strs=('minmax', "standard", "norm"),
                             match_method='contains', return_target_only=True)
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
        
    report['used_scikit_method']=use_sklearn
    report_obj= ReportFactory(title ="Data Scaling", **report )
    report_obj.add_mixed_types(report, table_width= 90,)
    return (data, report_obj) if return_report else data

def handle_outliers_in(
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
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import handle_outliers_in_data
    >>> data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [4, 5, 6, -50]})
    >>> data, report = handle_outliers_in_data(data, method='clip', view=True, 
                                               cmap='plasma', return_report=True)
    >>> print(report) 
    >>> report.lower_quantile
    0.01
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
        
    report_obj= ReportFactory(title ="Outliers Handling", **report )
    report_obj.add_mixed_types(report, table_width= 90,)
    return (data, report_obj) if return_report else data

@isdf
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
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import handle_missing_data
    >>> data = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
    >>> updated_data, report = handle_missing_data(
        data, view=True, method='fill_mean', return_report=True)
    >>> print(report) 
    >>> report.stats 
    {'method_used': 'fill_mean', 'fill_value': None, 'dropna_threshold': None}
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
        "describe%% Basic statistics": missing_data.describe()
    }
    report_obj= ReportFactory(title ="Missing Handling", **data_report )
    report_obj.add_mixed_types(data_report, table_width= 90)
    return (data, report_obj) if return_report else data

@isdf
def inspect_data(
    data: DataFrame, /, 
    correlation_threshold: float = 0.8, 
    categorical_threshold: float = 0.75, 
    include_stats_table=False, 
    return_report: bool=False
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
        
    include_stats_table: bool, default=False 
       If ``True`` include the table of the calculated statistic in the report. 
       Otherwise include the dictionnary values of basic statistics. 
       
    return_report: bool, default=False
        If set to ``True``, the function returns a ``Report`` object
        containing the comprehensive analysis of the data inspection. This
        report includes data integrity assessment, detailed statistics,
        and actionable recommendations for data preprocessing. This option
        provides programmatic access to the inspection outcomes, allowing
        further custom analysis or documentation. If ``False``, the function
        only prints the inspection report to the console without returning
        an object. This mode is suitable for interactive exploratory data
        analysis where immediate visual feedback is desired.
        
    Returns
    -------
    None or Report
        If `return_report` is set to `False`, the function prints a comprehensive
        report to the console, including assessments of data integrity, detailed
        statistics, and actionable recommendations for data preprocessing. This
        mode facilitates immediate, visual inspection of the data quality and
        characteristics without returning an object.
        
        If `return_report` is set to `True`, instead of printing, the function
        returns a `Report` object. This object encapsulates all findings from the
        data inspection, structured into sections that cover data integrity
        assessments, statistical summaries, and preprocessing recommendations.
        The `Report` object allows for programmatic exploration and manipulation
        of the inspection results, enabling users to integrate data quality
        checks into broader data processing and analysis workflows.
  
    Notes
    -----
    - The returned ``Report`` object is a dynamic entity providing structured
      access to various aspects of the data inspection process, such as
      integrity checks, statistical summaries, and preprocessing recommendations.
    - This feature is particularly useful for workflows that require
      a detailed examination and documentation of dataset characteristics
      and quality before proceeding with further data processing or analysis.
    - Utilizing the ``return_report`` option enhances reproducibility and
      traceability of data preprocessing steps, facilitating a transparent
      and accountable data analysis pipeline.
    
    Examples
    --------
    >>> from gofast.tools.dataops import inspect_data
    >>> import numpy as np
    >>> import pandas as pd
    
    Inspecting a DataFrame without returning a report object:
        
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['x', 'y', 'y']})
    >>> inspect_data(df)
    # Prints an exhaustive report to the console
    
    Inspecting a DataFrame and retrieving a report object for further analysis:
    
    >>> report = inspect_data(df, return_report=True)
    # A Report object is returned, enabling programmatic access to inspection details
    >>> print(report.integrity_report)
    # Access and print the integrity report part of the returned Report object
    
    >>> data = pd.DataFrame({
    >>>     'A': np.random.normal(0, 1, 100),
    >>>     'B': np.random.normal(5, 2, 100),
    >>>     'C': np.random.randint(0, 100, 100)
    >>> })
    >>> data.iloc[0, 0] = np.nan  # Introduce a missing value
    >>> data.iloc[1] = data.iloc[0]  # Introduce a duplicate row
    >>> report = inspect_data(data, return_report=True )
    >>> report.integrity_report
    <Report: Print to see the content>
    >>> report.integrity_report.outliers 
    {'A': 1, 'B': 0, 'C': 0}
    >>> report.stats_report
    Out[59]: 
        {'mean': A    -0.378182
         B     4.797963
         C    50.520000
         dtype: float64,
         'std_dev': A     1.037539
         B     2.154528
         C    31.858107
         dtype: float64,
         'percentiles':         0.25       0.50       0.75
         A  -1.072044  -0.496156   0.331585
         B   3.312453   4.481422   6.379643
         C  23.000000  48.000000  82.250000,
         'min': A   -3.766243
         B    0.054963
         C    0.000000
         dtype: float64,
         'max': A     2.155752
         B    10.751358
         C    99.000000
         dtype: float64}
    >>> report = inspect_data(data, include_stats_table=True, return_report=True)
    >>> report.stats_report
    <MultiFrame object with dataframes. Use print() to view.>

    >>> print(report.stats_report)
               Mean Values           
    =================================
               A       B        C
    ---------------------------------
    0    -0.1183  4.8666  48.3300
    =================================
            Standard Deviation       
    =================================
              A       B        C
    ---------------------------------
    0    0.9485  1.9769  29.5471
    =================================
               Percentitles          
    =================================
            0.25      0.5     0.75
    ---------------------------------
    A    -0.7316  -0.0772   0.4550
    B     3.7423   5.0293   5.9418
    C    18.7500  52.0000  73.2500
    =================================
              Minimum Values         
    =================================
               A        B       C
    ---------------------------------
    0    -2.3051  -0.1032  0.0000
    =================================
              Maximum Values         
    =================================
              A       B        C
    ---------------------------------
    0    2.3708  9.6736  99.0000
    =================================
    """  
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
    report ={}
    is_frame( data, df_only=True, raise_exception=True,
             objname="Data for inspection")
    is_valid, integrity_report = verify_data_integrity(data)
    stats_report = calculate_statistics(data)
    if stats_report and include_stats_table: 
        # contruct a multiframe objects from stats_report  
        stats_titles = ['Mean Values', 'Standard Deviation', 'Percentitles', 
                        'Minimum Values', 'Maximum Values' ]
        keywords, stats_data =  zip (*stats_report.items() )
        stats_report=  MultiFrameFormatter(stats_titles, keywords).add_dfs(
            *stats_data)
       
    report['integrity_status']= f"Checked ~ {integrity_report.integrity_checks}"
    report ['integrity_report']=integrity_report
    report ['stats_report'] = stats_report
    
    # Recommendations based on the report
    if not is_valid:
        report ["Recommendations"] = '-' *62
        
        if integrity_report['missing_values'].any():
            report['rec_missing_values']= (
                "- Consider handling missing values using imputation or removal."
                )
        if integrity_report['duplicates'] > 0:
            report['rec_duplicates']=(
                "- Check for and remove duplicate rows to ensure data uniqueness.")
        if any(count > 0 for count in integrity_report['outliers'].values()):
            report['rec_outliers']=(
                "- Investigate potential outliers. Consider removal or transformation.")
        
        # Additional checks and recommendations
        # Check for columns with a single unique value
        single_value_columns = [col for col in data.columns if 
                                data[col].nunique() == 1]
        if single_value_columns:
            report['rec_single_value_columns']=(
                "- Columns with a single unique value detected:"
                  f" {single_value_columns}. Consider removing them"
                  " as they do not provide useful information for analysis."
                  )
    
        # Check for data imbalance in categorical variables
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if data[col].value_counts(normalize=True).max() > categorical_threshold:
                report['rec_imbalance_data']=(
                    f"- High imbalance detected in categorical column '{col}'."
                    " Consider techniques to address imbalance, like sampling"
                    " methods or specialized models."
                    )
        # Check for skewness in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if abs(data[col].skew()) > 1:
                report['rec_skewness']=(
                    f"- High skewness detected in numeric column '{col}'."
                      " Consider transformations like log, square root, or "
                      "Box-Cox to normalize the distribution.")
        # Normalization for numerical columns
        report['normalization_evaluation']=(
            "- Evaluate if normalization (scaling between 0 and 1) is"
            " necessary for numerical features, especially for distance-based"
             " algorithms.")
        report['normalization_status'] = True 
        # Correlation check
        correlation_threshold = correlation_threshold  # Arbitrary threshold
        corr_matrix = data[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(
            corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col1, col2) for col1, col2 in zip(
            *np.where(upper_triangle > correlation_threshold))]
    
        report['correlation_checks'] = 'Passed' 
        if high_corr_pairs:
            report['high_corr_pairs'] =("- Highly correlated features detected:")
            for idx1, idx2 in high_corr_pairs:
                col1, col2 = numeric_cols[idx1], numeric_cols[idx2]
                report['rec_high_corr_pairs']=(
                    f"- {col1} and {col2} (Correlation > {correlation_threshold})")
        
        # Data type conversions
        report['rec_data_type_conversions'] = (
            "- Review data types of columns for appropriate conversions"
            " (e.g., converting float to int where applicable).")
    
    report_obj= ReportFactory(title ="Data Inspection", **report )
    report_obj.add_mixed_types(report, table_width= 90)
    
    if return_report: 
        return report_obj # return for retrieving attributes. 
    
    print(report_obj)
    
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
    >>> from gofast.tools.dataops import augment_data 
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

@isdf 
def prepare_data(
    data: DataFrame, 
    target_column: Optional [Union[str, List[str], Series, Array1D]]=None, 
    encode_categories: bool = False, 
    nan_policy: str='propagate', 
    verbose: bool = False, 
) -> Tuple[pd.DataFrame, Union[Series, DataFrame]]:
    """
    Prepare the feature matrix X and target vector y from the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The primary dataset from which features and target(s) are derived. 
        This DataFrame should include both the independent variables (features)
        and the dependent variable(s) (target). The structure and cleanliness 
        of this DataFrame directly affect the quality and reliability of the 
        output datasets (X and y). It is assumed that preprocessing steps like
        handling missing values, feature selection, and initial data cleaning 
        have been performed prior to this step, although some additional options
        for preprocessing are provided through other parameters of this function.
    
    target_column : Optional[Union[str, List[str], pd.Series, np.ndarray]] = None
        Specifies which columns within `data` represent the target variable(s) 
        for a predictive modeling task. This can be a single column name (str),
        a list of column names for multi-output tasks, or a separate data 
        structure (pd.Series or np.ndarray) containing the target variable(s). 
        If provided as a Series or ndarray, the length must match the number 
        of rows in `data`, aligning each observation with its corresponding 
        target value. The absence of this parameter (default None) implies that 
        the input data does not include target variables, and only feature 
        matrix X will be prepared. This flexibility allows the function to serve
        a variety of data preparation needs, from supervised learning (where 
        target variables are required) to unsupervised learning scenarios 
        (where no targets are necessary).
    
    encode_categories : bool = False
        A flag indicating whether to automatically convert categorical variables
        in `data` to a numerical format using LabelEncoder. Many machine learning
        models require numerical input and cannot handle categorical variables 
        directly. Setting this parameter to True automates the encoding process,
        transforming all object-type or category-type columns in `data` into 
        numerical format based on the unique values in each column. This encoding
        is unidirectional and primarily suitable for nominal categories without 
        a natural order. For ordinal categories or when more sophisticated 
        encoding strategies are required (e.g., one-hot encoding, frequency encoding),
        manual preprocessing is recommended. Note: enabling this option without 
        careful consideration of the nature of your categorical data can 
        introduce biases or inaccuracies in modeling.
    
    nan_policy : str = 'propagate'
        Determines how the function handles NaN (Not a Number) values in the 
        input DataFrame. This parameter can take one of three values:
        - 'propagate': NaN values are retained in the dataset. This option 
          does not alter the dataset regarding NaN values, implying that 
          downstream processing must handle NaNs appropriately.
        - 'omit': Rows or columns containing NaN values are removed from the 
          dataset before further processing. This option reduces the dataset 
          size but ensures that the remaining data is free from NaN values, 
          which might be beneficial for certain machine learning algorithms 
          that do not support NaN values.
        - 'raise': If NaN values are detected in the dataset, a ValueError is
          raised, halting execution. This policy is useful when the presence 
          of NaN values is unexpected and indicates a potential issue in the 
          data preparation or collection process that needs investigation.
    
    verbose : bool = False
        Controls the verbosity of the function's output. When set to True, the 
        function will print additional information about the preprocessing 
        steps being performed, such as the encoding of categorical variables or 
        the omission of NaN-containing entries. This can be helpful for debugging
        or understanding the function's behavior, especially during exploratory 
        data analysis phases or when integrating the function into larger data 
        processing pipelines.

    Returns
    -------
    X : pd.DataFrame**  
        A DataFrame containing the feature matrix with the target column(s) 
        removed if they were specified by name within the `data` DataFrame. 
        This matrix is ready for use in machine learning models as input variables.
    
    y : Union[pd.Series, pd.DataFrame]**  
        The target data extracted based on `target_column`. It is returned as 
        a pd.Series if a single target is specified or as a pd.DataFrame if 
        multiple targets are identified. For tasks where the target variable 
        is not defined (`target_column` is None), `y` is not returned.
    
    Raises
    ------
    ValueError  
        Raised in several scenarios:
        - If `target_column` is provided and its length does not match the 
          number of samples in `data`, ensuring data integrity and alignment 
          between features and targets.
        - If an invalid `nan_policy` is specified, ensuring that only the 
          predefined policies are applied to handle NaN values in the dataset.
        
    KeyError  
        Triggered if `target_column` is specified by name(s) and any of the 
        names are not found within the `data` DataFrame columns. This ensures 
        that the specified target variable(s) is actually present in the dataset.
    
    Examples
    --------
    Simple usage with automatic encoding of categorical features
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import prepare_data 
    >>> df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'target': [0, 1, 0]
    })
    >>> X, y = prepare_data(df, 'target', encode_categories=True)
    >>> print(X)
       feature1  feature2
    0         1         0
    1         2         1
    2         3         2
    >>> print(y)
       target
    0       0
    1       1
    2       0
       
    Handling NaN values by omitting them
    >>> df_with_nan = pd.DataFrame({
        'feature1': [np.nan, 2, 3],
        'feature2': ['A', np.nan, 'C'],
        'target': [1, np.nan, 0]
    })
    >>> X, y = prepare_data(df_with_nan, 'target', nan_policy='omit', verbose=True)
    "NaN values have been omitted from the dataset."
    
    Raising an error when NaN values are found
    >>> try:
    ...    prepare_data(df_with_nan, 'target', nan_policy='raise')
    ... except ValueError as e:
    ...    print(e)
       
    NaN values found in the dataset. Consider using 'omit' to drop them ...
  

    These examples illustrate the function's versatility in handling different 
    data preparation scenarios, from encoding categorical features for machine
    learning models to managing datasets with missing values according to 
    user-defined policies.
    """
    from sklearn.preprocessing import LabelEncoder
    
    nan_policy=str(nan_policy).lower()
    # Handle NaN values according to the specified policy
    texts = {}
    if nan_policy not in ['propagate', 'omit', 'raise']:
        raise ValueError("Invalid nan_policy. Choose from 'propagate', 'omit', 'raise'.")
    
    if nan_policy == 'raise' and data.isnull().values.any():
        raise ValueError("NaN values found in the dataset. Consider using 'omit' "
                         "to drop them or 'propagate' to ignore.")
    if nan_policy == 'omit':
        data = data.dropna().reset_index(drop=True)
        texts ['NaN status']= "NaN values have been omitted from the dataset." 
    
    if target_column is not None: 
        if isinstance(target_column, (str, list)):
            target_column= list(is_iterable(
                target_column, exclude_string= True, transform =True )
                )
            exist_features(data, features= target_column )
            y = data[target_column]
            X = data.drop(target_column, axis=1)
                
            y = data[target_column]
            X = data.drop(target_column, axis=1)
            
        elif isinstance(target_column, (pd.Series, np.ndarray)):
            if len(target_column) != len(data):
                raise ValueError("Length of the target array/Series must match "
                                 "the number of samples in 'data'.")
            y = pd.Series(target_column, index=data.index) if isinstance(
                target_column, np.ndarray) else target_column
            X = data
        else:
            raise ValueError("Invalid type for 'target_column'. Must be str, list,"
                             " pd.Series, or one-dimensional array.")
  
        check_consistent_length(X, y)
        
    else: X = data.copy () 
    # Check if there are categorical data columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
   
    if categorical_columns.any():
        # If encode_categories is False, raise a warning about unencoded categories
        msg= "Categorical features detected."
        
        texts ['categories_status']= msg 
        if not encode_categories:
            msg += (
                " It's recommended to encode categorical variables. Set"
                "`encode_categories=True` to automatically encode them."
                )
            texts ['categories_status']= msg # update 
            warnings.warn(msg, UserWarning)
        else:
            texts ['categories_status']= msg 
            # If encode_categories is True, encode the categorical features
            # label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                # Use LabelEncoder to transform the categorical columns
                X[col] = le.fit_transform(X[col])
                # Optionally, keep track of the label encoders for each column
                # label_encoders[col] = le

            texts['encoded_features']= ( 
                f"Encoded categorical feature(s): {list(categorical_columns)}.")
            
    if verbose and texts: 
        print(ReportFactory().add_recommendations(
            texts,  max_char_text= 90 ))
        
    return ( X, y ) if target_column is not None else X 

def assess_outlier_impact(
    data: DataFrame, /, 
    target_column: Union[str, List[str], Series, np.ndarray],
    test_size: float = 0.2, 
    random_state: int = 42, 
    encode_categories: bool=False, 
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
        
    encode_categories : bool, default False
        If True, categorical features in `data` are automatically encoded using 
        LabelEncoder. This is useful for models that require numerical input for
        categorical data.
    
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
    >>> from gofast.tools.dataops import assess_outlier_impact
    >>> df = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })
    >>> mse_with_outliers, mse_without_outliers = assess_outlier_impact(df, 'target')

    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    is_frame (data, df_only= True, raise_exception= True )
    X, y = prepare_data(
        data, target_column, encode_categories=encode_categories,
        nan_policy= 'omit' )
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
        model.fit(X_train, y_train.ravel() )
        predictions = model.predict(X_test)
        return metric(y_test.ravel(), predictions)

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
    filtered_metric = _evaluate_model(X_train_filtered, y_train_filtered,
                                      X_test, y_test, model, metric)

    # Print results if verbose is True
    if verbose:
        texts ={'original_metric_status': (
            f'{metric_name} with outliers in the training set: {original_metric}'), 
            'filtered_metric_status': (
                f'{metric_name} without outliers in the training set: {filtered_metric}')
            }
        # Check the impact
        if not is_categorical and filtered_metric < original_metric or \
           is_categorical and filtered_metric > original_metric:
            texts['outliers_impact']= (
                'Outliers appear to have a negative impact on the model performance.')
        else:
            texts['outliers_impact']= ('Outliers do not appear to have a significant negative'
                  ' impact on the model performance.')
            
        print(ReportFactory().add_recommendations(texts,  max_char_text= 90 ))

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
    >>> from gofast.tools.dataops import transform_dates
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
    >>> from gofast.tools.dataops import merge_frames_on_index
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
    >>> from gofast.tools.dataops import apply_tfidf_vectorization
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
    >>> from gofast.tools.dataops import apply_bow_vectorization
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

@Dataify 
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
    >>> from gofast.tools.dataops import apply_word_embeddings
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

@ensure_pkg("gensim","Word-Embeddings expect 'gensim'to be installed." )
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
    >>> from gofast.tools.dataops import boxcox_transformation
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
    is_frame (data, df_only=True, raise_exception= True,
              objname='Boxcox transformation' )
    
    transformed_data = pd.DataFrame()
    lambda_values = {}
    verbosity_texts = {}

    if columns is not None:
        columns = is_iterable(columns, exclude_string=True, transform= True )
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            verbosity_texts['missing_columns']=(
                f" Columns {missing_cols} not found in DataFrame."
                 " Skipping these columns.")
        columns = [col for col in columns if col in data.columns]

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_transform = columns if columns is not None else numeric_columns
    verbosity_texts['columns_to_transform']=(
        f"Transformed columns: {list(columns_to_transform)}")
    
    # validate adjust_non_positive parameter
    adjust_non_positive=parameter_validator(
        "adjust_non_positive", ["adjust", "skip"], 
        error_msg= (
                "`adjust_non_positive` argument expects ['skip', 'adjust']"
               f" Got: {adjust_non_positive!r}")
        ) (adjust_non_positive)

    skipped_num_columns=[]
    skipped_non_num_columns =[]
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
                skipped_num_columns.append (column)
                
            transformed_data[column] = transformed
            lambda_values[column] = fitted_lambda
        else:
            skipped_non_num_columns.append (column)
            
    if skipped_num_columns: 
        verbosity_texts['skipped_columns']=(
            f"Column(s) '{skipped_num_columns}' skipped: contains values <= {min_value}.")
    if skipped_non_num_columns: 
        verbosity_texts['non_numeric_columns']=(
            f"Column(s) '{skipped_non_num_columns}' is not numeric and will be skipped.")

    # Include non-transformed columns in the returned DataFrame
    for column in data.columns:
        if column not in transformed_data:
            transformed_data[column] = data[column]
            
    if view:
        # Initialize a flag to check the presence of valid data for heatmaps
        valid_data_exists = True
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0] * 2, fig_size[1]))
        
        # Determine columns suitable for the heatmap
        heatmap_columns = columns if columns else data.select_dtypes(
            include=[np.number]).columns
        heatmap_columns = [col for col in heatmap_columns if col in data.columns 
                           and np.issubdtype(data[col].dtype, np.number)]
        
        # Check for non-NaN data in original dataset and plot heatmap
        if heatmap_columns and not data[heatmap_columns].isnull().all().all():
            sns.heatmap(data[heatmap_columns].dropna(axis=1, how='all').corr(), ax=axs[0],
                        annot=True, cmap=cmap)
            axs[0].set_title('Correlation Matrix Before Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['before_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix Before Transformation') 
        
        # Verify transformed_data's structure and plot its heatmap
        if 'transformed_data' in locals() and not transformed_data[heatmap_columns].isnull(
                ).all().all():
            sns.heatmap(transformed_data[heatmap_columns].dropna(
                axis=1, how='all').corr(), ax=axs[1], annot=True, cmap=cmap)
            axs[1].set_title('Correlation Matrix After Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['after_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix After Transformation') 
        
        # Display the plots if valid data exists; otherwise, close the plot
        # to avoid displaying empty figures
        if valid_data_exists:
            plt.tight_layout()
            plt.show()
        else:
            verbosity_texts['matplotlib_window_status']='Closed'
            plt.close()  # Closes the matplotlib window if no valid data is present
    
    # Print verbose recommendations if any and if verbose mode is enabled
    if verbose and verbosity_texts:
        recommendations = ReportFactory('BoxCox Transformation').add_recommendations(
            verbosity_texts, max_char_text=90)
        print(recommendations)
        
    return transformed_data, lambda_values

@isdf 
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
    >>> from gofast.tools.dataops import check_missing_data
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
    verbosity_texts={}
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
            verbosity_texts['start_angle']=(
                "Start angle greater than 180 degrees. Using modulo "
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
        
    verbosity_texts ['missing_stats%% Missing Table']= missing_stats
    if verbose and verbosity_texts: 
        summary = ReportFactory('Missing Report').add_mixed_types(
            verbosity_texts, table_width= 70 )
        print(summary)

    return missing_stats

@DataTransformer('data', mode='lazy')
@DynamicMethod(
    'both', 
    capture_columns=True, 
    prefixer='exclude' 
   )
def base_transform(
    data: DataFrame, 
    target_columns:Optional[str|List[str]]=None, 
    columns: Optional[str|List[str]]=None, 
    noise_level: float=None, 
    seed: int=None):
    """
    Applies preprocessing transformations to the specified DataFrame, including 
    handling of missing values, feature scaling, encoding categorical variables, 
    and optionally introducing noise to numeric features. Transformations can 
    be selectively applied to specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to undergo preprocessing transformations.
    target_columns : str or list of str, optional
        The name(s) of the column(s) considered as target variable(s). 
        These columns are excluded from transformations to prevent leakage. 
        Default is None, indicating
        no columns are treated as targets.
    columns : str or list of str, optional
        Specific columns to which transformations should be applied. 
        If None, transformations
        are applied to all columns excluding `target_columns`. 
        Default is None.
    noise_level : float, optional
        The level of noise (as a fraction between 0 and 1) to introduce into 
        the numeric columns. If None, no noise is added. Default is None.
    seed : int, optional
        Seed for the random number generator, ensuring reproducibility of the noise
        and other random aspects of preprocessing. Default is None.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame with numeric features scaled, missing values imputed,
        categorical variables encoded, and optionally noise added.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification

    # Generating a synthetic dataset
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3',
    ...                                 'feature_4'])
    >>> data['target'] = y

    # Apply base_transform to preprocess features, excluding the target column
    >>> from gofast.tools.dataops import base_transform 
    >>> preprocessed_data = base_transform(data, target_columns='target', 
    ...                                    noise_level=0.1, seed=42)
    >>> print(preprocessed_data.head())

    Note
    ----
    This function is designed to be flexible, allowing selective preprocessing
    on parts of the DataFrame. It leverages `ColumnTransformer` to efficiently 
    process columns based on their data type. The inclusion of `noise_level` 
    allows for simulating real-world data imperfections.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    np.random.seed(seed)
    target_columns = [target_columns] if isinstance(
        target_columns, str) else target_columns

    if target_columns is not None:
        data = data.drop(columns=target_columns, errors='ignore')
    # get original data columns 
    original_columns = list(data.columns  )
    # Identify numeric and categorical features for transformation
    numeric_features = data.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()

    # Define transformations for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Apply ColumnTransformer to handle each feature type
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    data_processed = preprocessor.fit_transform(data)
    processed_columns = numeric_features + categorical_features 
    try: 
        data_processed = pd.DataFrame(
            data_processed, columns=processed_columns,  
            index=data.index)
    except : 
        data_processed = pd.DataFrame(
            data_processed, index=data.index)

    # Apply noise to non-target numeric columns if specified
    if noise_level is not None:
        noise_level = assert_ratio ( noise_level )
        # Add noise if specified
        # if  noise_level > 0:
        #     noise_mask = np.random.rand(*data_processed.shape) < noise_level
        #     data_processed.where(~noise_mask, other=np.nan, inplace=True)
        assert 0 <= noise_level <= 1, "noise_level must be between 0 and 1"
        for column in numeric_features:
            noise_mask = np.random.rand(data.shape[0]) < noise_level
            data_processed.loc[noise_mask, column] = np.nan
    # now rearrange the columns back 
    try:
        return  data_processed [original_columns ]
    except:
        return data_processed # if something wrong, return it
 
@Dataify(auto_columns= True , ignore_mismatch=True, prefix="var_")
def analyze_data_corr(
    data: DataFrame, 
    columns: Optional[ List[str]]=None, 
    method: str | Callable [[ArrayLike, ArrayLike], float]='pearson', 
    min_periods: int=1, 
    min_corr: float =0.5, 
    high_corr: float=0.8, 
    use_symbols: bool =False, 
    hide_diag: bool =True,
    no_corr_placeholder: str='...', 
    view: bool = False,
    cmap: str = 'viridis', 
    fig_size: Tuple[int, int] = (8, 8)
    ):
    """
    Computes the correlation matrix for specified columns in a pandas DataFrame
    and optionally visualizes it using a heatmap. 
    
    This function can also symbolically represent correlation values and 
    selectively hide diagonal elements in the visualization and interpretability.

    Parameters
    ----------
    data : DataFrame
        The DataFrame from which to compute the correlation matrix.
    columns : Optional[List[str]], optional
        Specific columns to consider for the correlation calculation. If None, all
        numeric columns are used. Default is None.
    method : str | Callable[[ArrayLike, ArrayLike], float], optional
        Method to use for computing the correlation:
        - 'pearson' : Pearson correlation coefficient
        - 'kendall' : Kendall Tau correlation coefficient
        - 'spearman' : Spearman rank correlation
        - Custom function : a callable with input of two 1d ndarrays and returning a float
        Default is 'pearson'.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to have a valid result. 
        Default is 1.
    min_corr : float, optional
        The minimum threshold for correlations to be noted if using symbols. 
        Default is 0.5.
    high_corr : float, optional
        Threshold above which correlations are considered high, relevant if
        `use_symbols` is True. Default is 0.8.
    use_symbols : bool, optional
        Whether to use symbolic representation ('++', '--', '+-') instead of 
        numeric values where: 
        - ``'++'``: Represents a strong positive relationship.
        - ``'--'``: Represents a strong negative relationship.
        - ``'-+'``: Represents a moderate relationship.
        - ``'o'``: Used exclusively for diagonal elements, typically representing
          a perfect relationship in correlation matrices (value of 1.0).
        Default is False.
    hide_diag : bool, optional
        If True, diagonal values in the correlation matrix visualization are hidden.
        Default is True.
    no_corr_placeholder : str, optional
        Text to display for correlation values below `min_corr`. Default is '...'.
    view : bool, optional
        If True, displays a heatmap of the correlation matrix using matplotlib and
        seaborn. Default is False.
    cmap : str, optional
        The colormap for the heatmap visualization. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Dimensions of the figure that displays the heatmap. Default is (8, 8).

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the correlation matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import analyze_data_corr
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [2, 3, 4, 5, 6]
    ... })
    >>> corr_summary = analyze_data_corr(data, view=True)
    >>> corr_summary
    <Summary: Populated. Use print() to see the contents.>

    >>> print(corr_summary) 
          Correlation Table       
    ==============================
            A        B        C   
      ----------------------------
    A |           -1.0000   1.0000
    B |  -1.0000           -1.0000
    C |   1.0000  -1.0000         
    ==============================
    
    >>> corr_summary.corr_matrix
         A    B    C
    A  1.0 -1.0  1.0
    B -1.0  1.0 -1.0
    C  1.0 -1.0  1.0
    
    >>> corr_summary = analyze_data_corr(data, view=False, use_symbols=True)
    >>> print(corr_summary) 
      Correlation Table  
    =====================
          A     B     C  
      -------------------
    A |        --    ++  
    B |  --          --  
    C |  ++    --        
    =====================

    .....................
    Legend : ...:
             Non-correlated,
             ++: Strong
             positive, --:
             Strong
             negative, -+:
             Moderate
    .....................
    
    Notes
    -----
    If `view` is True, the function requires a matplotlib backend that supports
    interactivity, typically within a Jupyter notebook or a Python environment
    configured for GUI operations.
    """
    numeric_df = data.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric data found in the DataFrame.")
        
    # Compute the correlation matrix
    method = method.lower() if isinstance (method, str) else method  
    correlation_matrix = numeric_df.corr(method = method, min_periods = min_periods)
 
    # Check if the user wants to view the heatmap
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
        plt.title('Heatmap of Correlation Matrix')
        plt.show()
        
    summary = Summary(corr_matrix=correlation_matrix )
    summary.add_data_corr(
        correlation_matrix, 
        min_corr=assert_ratio( min_corr, bounds=(0, 1)),
        high_corr=assert_ratio(high_corr, bounds=(0, 1)), 
        use_symbols= use_symbols, 
        hide_diag= hide_diag,
        precomputed=True,
        no_corr_placeholder=str(no_corr_placeholder)
        )
    return summary

@Dataify(auto_columns=True)
def data_assistant(data: DataFrame, view: bool=False):
    """
    Performs an in-depth analysis of a pandas DataFrame, providing insights,
    identifying data quality issues, and suggesting corrective actions to
    optimize the data for statistical modeling and analysis. The function 
    generates a report that includes recommendations and possible actions 
    based on the analysis.

    The function performs a comprehensive analysis on the DataFrame, including
    checks for data integrity, descriptive statistics, correlation analysis, 
    and more, `data_assistant` provides recommendations for each identified 
    issue along with suggestions for modules and functions that could be 
    used to resolve these issues. In otherwords, function is meant to assist 
    users in understanding their data better and preparing it for further 
    modeling steps.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be analyzed. This function expects a pandas DataFrame
        populated with data that can be numerically or categorically analyzed.

    view : bool, optional
        A boolean flag that, when set to True, enables the visualization of
        certain data aspects such as distribution plots and correlation matrices.
        Defaults to False, which means no visualizations are displayed.

    Returns
    -------
    None
        Function does not return any value; instead, it prints a detailed
        report directly to the console. The report includes sections on data
        integrity, descriptive statistics, and potential issues along with
        suggestions for data preprocessing steps.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import data_assistant
    >>> df = pd.DataFrame({
    ...     'Age': [25, 30, 35, 40, None],
    ...     'Salary': [50000, 60000, 70000, 80000, 90000],
    ...     'City': ['New York', 'Los Angeles', 'San Francisco', 'Houston', 'Seattle']
    ... })
    >>> data_assistant(df)

    Notes
    -----
    The `data_assistant` function is designed to assist in the preliminary analysis
    phase of data processing, offering a diagnostic view of the data's quality and
    structure. It is particularly useful for identifying and addressing common
    data issues before proceeding to more complex data modeling or analysis tasks.

    It is recommended to review the insights and apply the suggested transformations
    or corrections to ensure the data is optimally prepared for further analysis
    or machine learning modeling.

    See Also
    --------
    gofast.tools.dataops.inspect_data : A related function that provides a deeper
    dive into data inspection with more customizable features.
    
    """
    # Get the current datetime
    current_datetime = datetime.datetime.now()
    # Format the datetime object into a string in the specified format
    formatted_datetime = current_datetime.strftime("%d %B %Y %H:%M:%S")
    # Creating a dictionary with the formatted datetime string
    texts = {"Starting assistance...": formatted_datetime}
    
    # Initialize dictionaries to store sections of the report
    recommendations = {"RECOMMENDATIONS": "-" * 12}
    helper_funcs = {"HELPER FUNCTIONS": "-" * 12}
    
    # Checking for missing values
    texts ["1. Checking for missing values"]="Passed"
    if data.isnull().sum().sum() > 0:
        texts ["1. Checking for missing values"]="Failed"
        texts["   #  Found missing values in the dataset?"]='yes'
        #print("Found missing values in the dataset.")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        texts[ "   #  Columns with missing values and counts"]= smart_format(
            [missing_data.name] if isinstance (
                missing_data, pd.Series ) else missing_data.columns ) 
        
        recommendations["1. rec-missing values "]= ( 
            "Missing data can lead to biased or invalid conclusions"
            " if not handled properly, as many statistical methods assume"
            " complete data. Consider imputing or dropping the missing values."
            " See helper functions for handling missing values."
            ) 
        helper_funcs["1. help-missing values "]= ( 
            "Use: pandas.DataFrame.fillna(), `sklearn.impute.SimpleImputer`"
            " `gofast.tools.soft_imputer`, `gofast.tools.one_click_preprocess`"
            " `gofast.tools.handle_missing_values` and more..."
            )
    # Descriptive statistics
    texts ["2. Descriptive Statistics"]="Done"
    texts["   #  summary"]= format_iterable(data.describe())
    # Check for zero variance features
    texts ["3. Checking zero variance features"]="Passed"
    zero_var_cols = data.columns[data.nunique() == 1]
    if len(zero_var_cols) > 0:
        texts ["3. Checking zero variance features"]="Failed"
        texts ["   #  Found zero variance columns?"]="yes"
        texts["   #  Zeros variances columns"]=( 
            f"{smart_format(zero_var_cols.tolist())}"
            )
        
        recommendations["3. rec-zero variances features"]=(
            "Zero variance features offer no useful information for modeling"
            " because they do not vary across observations. This means they"
            " cannot help in predicting the target variable or in distinguishing"
            " between different instances. Consider dropping them as they do not "
            " provide any information, redundant computation, model complexity..."
            )
        helper_funcs["3. help-zero variances features"]= ( 
            "Use: `pandas.DataFrame.drop(columns =<zero_var_cols>)`")
        
    # Data types analysis
    texts["4. Data types summary"]="Passed"
    if (data.dtypes == 'object').any():
        texts["   #  Summary types"]="Include string or mixed types"
        texts["   #  Non-numeric data types found?"]="yes"
        texts["   #  Non-numeric features"]= smart_format( 
            data.select_dtypes( exclude=[np.number]).columns.tolist())
        
        recommendations [ "4. rec-non-numeric data"]= (
            "Improper handling of non-numeric data can lead to misleading"
            " results or poor model performance. For instance, ordinal "
            " encoding of inherently nominal data can imply a nonexistent"
            " ordinal relationship, potentially misleading the model."  
            " Consider transforming into a numeric format through encoding"
            " techniques (like one-hot encoding, label encoding, or embedding)"
            " to be used in these models.")
        helper_funcs ["4. help-non-numeric data"]=( 
            "Use: `pandas.get_dummies()`, `sklearn.preprocessing.LabelEncoder`"
            " `gofast.tools.codify_variables`,"
            " `gofast.tools.handle_categorical_features` and more ..."
            ) 
        
    # Correlation analysis
    texts["5. Correlation analysis"]="Passed"
    numeric_data = data.select_dtypes(include=[np.number])
    texts["   #  Numeric corr-feasible features"]=format_iterable(numeric_data)
    if numeric_data.shape[1] > 1:
 
        texts["   #  Correlation matrix review"]="yes"
        if view: 
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()
            texts["   #  corr-matrix view"]="See displayed figure..."
            
        recommendations["5. rec-correlated features"]= (
            "Highly correlated features can lead to multicollinearity in"
            " regression models, where it becomes difficult to estimate the"
            " relationship of each independent variable with the dependent"
            " variable due to redundancy. Review highly correlated variables"
            " as they might affect model performance due to multicollinearity."
            )
        helper_funcs ["5. help-correlated features"]= ( 
            "Use: `pandas.DataFrame.go_corr`, `gofast.tools.analyze_data_corr`,"
            " gofast.tools.correlation_ops``gofast.drop_correlated_features`,"
            " `gofast.stats.corr` and more ...")
        
    # Distribution analysis
    texts["6. Checking for potential outliers"]="Passed"
    skew_cols =[]
    for col in numeric_data.columns:
        if view: 
            plt.figure()
            sns.distplot(data[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()
           
        if data[col].skew() > 1.5 or data[col].skew() < -1.5:
            skew_value = data[col].skew()
            skew_cols.append ( (col, skew_value) ) 
            
    if view: 
        texts["   #  Distribution analysis view"]="See displayed figure..."
        
    if skew_cols : 
        texts["   #  Outliers found?"]="yes"
        texts["   #  Skewness columns"]=', '.join(
            [ f"{skew}-{val:.4f}" for skew, val in  skew_cols ]) 
        recommendations["6. rec-distribution ~ skewness"]= (
            "Skewness can distort the mean and standard deviation of the data."
            " Measures of skewed data do not accurately represent the center"
            " and variability of the data, potentially leading to misleading"
            " interpretations, poor model performance and unreliable predictions."
            " Consider transforming this data using logarithmic, square root,"
            " or box-cox transformations")
        helper_funcs ["6. help-distribution ~ skewness" ]= ( 
            "Use: `scipy.stats.boxcox`, `sklearn.preprocessing.PowerTransformer`"
            " `gofast.tools.handle_skew`, `gofast.tools.assess_outlier_impact`"
            " `pandas.DataFrame.go_skew`, `gofast.stats.skew` and more ..."
            )
    # Duplicate rows
    texts["7. Duplicate analysis"]="Passed"
    if data.duplicated().sum() > 0:
        texts["   #  Found duplicate rows in the dataset?"]="yes"
        texts["   #  Duplicated indices"]=smart_format(
            handle_duplicates(data, return_indices=True )
            ) 
        recommendations["7. rec-duplicate analysis"]=(
            "Duplicate entries can skew statistical calculations such as means,"
            " variances, and correlations, leading to biased or incorrect estimates"
            " Duplicate rows can lead to overfitting, particularly if the"
            " duplicates are only present in the training dataset and not in"
            " unseen data. Models might learn to overly rely on these repeated"
            " patterns, performing well on training data but poorly on new,"
            " unseen data. Consider reviewing and possibly removing duplicates."
            )
        helper_funcs ["7. help-duplicate analysis"]=(
            "Use: `pandas.DataFrame.drop_duplicates()`,"
            " `gofast.tools.handle_duplicates` and more ...")
        
    # Unique value check
    texts["8. Unique value check"]="Passed"
    unique_cols =[]
    for col in data.columns:
        if data[col].nunique() < 10:
            value = data[col].unique()
            unique_cols.append (( col, list(value))) 
    
    if unique_cols: 
        #print(unique_cols)
        texts["   #  Found unique value column?"]="yes"
        texts["   #  Duplicated indices"]=', '.join(
            [ f"{col}-{str(val) if len(val)<=3 else format_iterable(val)}" 
             for col, val in unique_cols ]
            ) 
        recommendations ["8. rec-Unique identifiers"]= ( 
            "Unique identifiers typically do not possess any intrinsic"
            " predictive power because they are unique to each record."
            " Including these in predictive modeling can be misleading"
            " for algorithms, leading them to overfit the data without"
            " gaining any generalizable insights. Check if these columns"
            " should be treated as a categorical variables"
            )
        helper_funcs["8. help-unique identifiers"]= ( 
            "Use: `gofast.tools.handle_unique_identifiers`,"
            " `gofast.transformers.BaseCategoricalEncoder`,"
            " `gofast.transformer.CategoricalEncoder`, and more ..."
            )
    
    report_txt= {**texts, **recommendations, **helper_funcs}
 
    # Check if the recommendations and helper functions 
    # are empty (only contain the header)
    if len(recommendations) == 1:
        # Remove the "RECOMMENDATIONS" title from 
        # the report if no recommendations exist
        report_txt.pop("RECOMMENDATIONS")
    
    if len(helper_funcs) == 1:
        # Remove the "HELPER FUNCTIONS" title from the report 
        # if no helper functions exist
        report_txt.pop("HELPER FUNCTIONS")
    
    # In case both sections are empty, append a general note
    if len(recommendations) == 1 and len(helper_funcs) == 1:
        # Adding a general keynote to the report to address the absence
        # of recommendations and helper functions
        report_txt["KEYNOTE"] = "-" * 12
        report_txt["TO DO"] = (
            "Review the provided insights. No further immediate actions"
            " or recommendations are required at this stage. For a more"
            " in-depth inspection or specific queries, consider using"
            " detailed analysis tools such as `gofast.tools.inspect_data`."
        )
    else:
        # Provide actionable steps when there are recommendations
        # and/or helper functions
        report_txt["NOTE"] = "-" * 12
        report_txt["TO DO"] = (
            "Review the insights and recommendations provided to effectively"
            " prepare your data for modeling. For more in-depth analysis,"
            " utilize tools such as `gofast.tools.inspect_data`."
        )

    assistance_report = ReportFactory(" Data Assistant Report").add_mixed_types(
        report_txt, table_width= 100  )
    
    print(assistance_report)

def correlation_ops(
    data: DataFrame, 
    correlation_type:str='all', 
    min_corr: float=0.5, 
    high_corr: float=0.8, 
    method: str| Callable[[ArrayLike, ArrayLike], float]='pearson', 
    min_periods: int=1, 
    display_corrtable: bool=False, 
    **corr_kws
    ):
    """
    Performs correlation analysis on a given DataFrame and classifies the 
    correlations into specified categories. Depending on the `correlation_type`,
    this function can categorize correlations as strong positive, strong negative,
    or moderate. It can also display the correlation matrix and returns a 
    formatted report of the findings.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame on which to perform the correlation analysis.
    correlation_type : str, optional
        The type of correlations to consider in the analysis. Valid options
        are 'all', 'strong only', 'strong positive', 'strong negative', and
        'moderate'. Defaults to 'all'.
    min_corr : float, optional
        The minimum correlation value to consider for moderate correlations.
        Defaults to 0.5.
    high_corr : float, optional
        The threshold above which correlations are considered strong. Defaults
        to 0.8.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Method of correlation:
        - 'pearson' : standard correlation coefficient
        - 'kendall' : Kendall Tau correlation coefficient
        - 'spearman' : Spearman rank correlation
        Defaults to 'pearson'.
    min_periods : int, optional
        Minimum number of observations required per pair of columns to have a
        valid result. Defaults to 1.
    display_corrtable : bool, optional
        If True, prints the correlation matrix to the console. Defaults to False.
    **corr_kws : dict
        Additional keyword arguments to be passed to the correlation function
        :func:`analyze_data_corr`. 

    Returns
    -------
    MultiFrameFormatter or ReportFactory
        Depending on the analysis results, returns either a formatted report of 
        correlation pairs or a message about the absence of significant 
        correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import correlation_ops
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 2, 3, 4, 4],
    ...     'C': [5, 3, 2, 1, 1]
    ... })
    >>> result = correlation_ops(data, correlation_type='strong positive')
    >>> print(result)

    Notes
    -----
    The correlation threshold parameters (`min_corr` and `high_corr`) help in 
    fine-tuning which correlations are reported based on their strength. This 
    is particularly useful in datasets with many variables, where focusing on 
    highly correlated pairs is often more insightful.

    See Also
    --------
    pandas.DataFrame.corr : Compute pairwise correlation of columns.
    MultiFrameFormatter : A custom formatter for displaying correlated pairs.
    analyze_data_corr : A predefined function for computing and summarizing 
                      correlations.
    """

    # Compute the correlation matrix using a predefined analysis function
    corr_summary = analyze_data_corr(
        data, method=method, min_periods=min_periods, **corr_kws)
    
    if display_corrtable:
        print(corr_summary)
    
    corr_matrix = corr_summary.corr_matrix
    # validate correlation_type parameter 
    correlation_type = parameter_validator('correlation_type',
        ["all","strong only", "strong positive", "strong negative", "moderate" ]
        )(correlation_type)
    # Storage for correlation pairs
    strong_positives, strong_negatives, moderates = [], [], []

    # Evaluate each cell in the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if ( 
                    correlation_type in ['all', 'strong only', 'strong positive'] 
                    and corr_value >= high_corr ) :
                strong_positives.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_value)
                    )
            if ( 
                    correlation_type in ['all', 'strong only', 'strong negative'] 
                    and corr_value <= -high_corr ):
                strong_negatives.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_value)
                    )
            if ( 
                    correlation_type in ['all', 'moderate'] 
                    and min_corr <= abs(corr_value) < high_corr ) :
                moderates.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                  corr_value))

    # Prepare DataFrames for each category
    dfs = {}
    if strong_positives:
        dfs['Strong Positives'] = pd.DataFrame(
            strong_positives, columns=['Feature 1', 'Feature 2', 'Correlation'])
    if strong_negatives:
        dfs['Strong Negatives'] = pd.DataFrame(
            strong_negatives, columns=['Feature 1', 'Feature 2', 'Correlation'])
    if moderates:
        dfs['Moderates'] = pd.DataFrame(
            moderates, columns=['Feature 1', 'Feature 2', 'Correlation'])

    # Formatting the output with MultiFrameFormatter if needed
    if dfs:
        formatted_report = MultiFrameFormatter(list(dfs.keys())).add_dfs(*dfs.values())
        setattr(formatted_report, "correlated_pairs", dfs)
        return formatted_report
    else:
        insights=ReportFactory(title=f"Correlation Type: {correlation_type}"
                               ).add_recommendations(
            (
            "No significant correlations detected in the provided dataset. "
            "This may indicate that data variables act independently of each other "
            "within the specified thresholds. Consider adjusting the correlation "
            "thresholds or analyzing the data for other patterns."
            ), 
            keys= 'Actionable Insight', max_char_text= 90
            )
        return insights
    
@Dataify (auto_columns=True)    
def drop_correlated_features(
    data: DataFrame, 
    method: str | Callable[[ArrayLike, ArrayLike], float] = 'pearson', 
    threshold: float = 0.8, 
    display_corrtable: bool = False, 
    **corr_kws
    ):
    """
    Analyzes and removes highly correlated features from a DataFrame to reduce 
    multicollinearity, improving the reliability and performance of subsequent 
    statistical models. This function allows for customization of the correlation 
    computation method and the threshold for feature removal.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame from which correlated features are to be removed.
    method : {'pearson', 'kendall', 'spearman'} or callable, optional
        Method of correlation to be used. The method can be one of 'pearson' 
        (default), 'kendall', 'spearman', or a callable with the signature 
        Callable[[ArrayLike, ArrayLike], float] providing a custom correlation 
        computation. The default is 'pearson', which assesses linear 
        relationships.
    threshold : float, optional
        The correlation coefficient threshold above which one of the features 
        in a pair will be removed. Defaults to 0.8, where features with a 
        correlation coefficient higher than this value are considered highly 
        correlated.
    display_corrtable : bool, optional
        If set to True, the correlation matrix is printed before removing 
        correlated features. This can be useful for visualization and manual 
        review of the correlation values. Defaults to False.
    **corr_kws : dict
        Additional keyword arguments to be passed to the 
        :func:`analyze_data_corr` correlation function.

    Returns
    -------
    pandas.DataFrame
        Returns a DataFrame with the highly correlated features removed based 
        on the specified threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import drop_correlated_features
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 2, 3, 4, 4],
    ...     'C': [5, 3, 2, 1, 1]
    ... })
    >>> print(data.corr())
    >>> reduced_data = drop_correlated_features(data, threshold=0.8)
    >>> print(reduced_data)

    Notes
    -----
    Removing correlated features is a common preprocessing step to avoid 
    multicollinearity, which can distort the estimates of a model's parameters 
    and affect the interpretability of variable importance. This function 
    is particularly useful in the preprocessing steps for statistical modeling 
    and machine learning.

    The choice of correlation method and threshold should be guided by specific 
    analytical needs and the nature of the dataset. Lower thresholds increase 
    the number of features removed, potentially simplifying the model but at 
    the risk of losing important information.

    See Also
    --------
    pandas.DataFrame.corr : Method to compute pairwise correlation of columns.
    gofast.tools.dataops.analyze_data_corr: Function to analyze correlations 
    with more detailed options and outputs.
    """
    # Compute the correlation matrix using a predefined analysis function
    corr_summary = analyze_data_corr(
        data, method=method, high_corr=threshold,  **corr_kws)
    
    if display_corrtable:
        print(corr_summary)
        
    # Compute the absolute correlation matrix and the upper triangle
    corr_matrix = corr_summary.corr_matrix.abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify columns to drop based on the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop the identified columns and return the reduced DataFrame
    df_reduced = data.drop(to_drop, axis=1)
    
    return df_reduced

@Dataify (auto_columns=True)
def handle_skew(data: DataFrame, method: str = 'log'):
    """
    Applies a specified transformation to numeric columns in the DataFrame 
    to correct for skewness. This function supports logarithmic, square root,
    and Box-Cox transformations, helping to normalize data distributions and 
    improve the performance of many statistical models and machine learning 
    algorithms.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing numeric data that may exhibit skewness.
    method : {'log', 'sqrt', 'box-cox'}, optional
        The method of transformation to apply:
        - 'log' : Applies the natural logarithm transformation. Suitable for data
                  that is positively skewed. Cannot be applied to zero or 
                  negative values.
        - 'sqrt': Applies the square root transformation. Suitable for reducing 
                  moderate skewness. Cannot be applied to negative values.
        - 'box-cox': Applies the Box-Cox transformation which can handle broader
                     ranges and types of skewness but requires all data points to be
                     positive. If any data points are not positive, a generalized
                     Yeo-Johnson transformation (handled internally) is applied
                     instead.
        Default is 'log'.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with transformed data to address skewness in numeric 
        columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import handle_skew
    >>> data = pd.DataFrame({
    ...     'A': [0.1, 1.5, 3.0, 4.5, 10.0],
    ...     'B': [-1, 2, 5, 7, 9]
    ... })
    >>> transformed_data = handle_skew(data, method='log')
    >>> print(transformed_data)

    Notes
    -----
    Skewness in a dataset can lead to biases in machine learning and statistical 
    models, especially those that assume normality of the data distribution. By 
    transforming skewed data, this function helps mitigate such issues, enhancing 
    model accuracy and robustness.

    It is important to understand the nature of your data and the requirements of
    your specific models when choosing a transformation method. Some methods, like
    'log', cannot handle zero or negative values without adjustments.

    See Also
    --------
    scipy.stats.boxcox : For more details on the Box-Cox transformation.
    sklearn.preprocessing.PowerTransformer : Implements both the Box-Cox transformation
                                             and the Yeo-Johnson transformation.
    """
    from sklearn.preprocessing import PowerTransformer

    # Validate and apply the chosen method to each numeric column
    method = parameter_validator('method', ["log", "sqrt", "box-cox"])(method)
    # validate skew method 
    [validate_skew_method(data[col], method) for col in data.columns ]
    for column in data.select_dtypes(include=['int64', 'float64']):
        # Adjust for non-positive values where necessary
        if data[column].min() <= 0:  
            data[column] += (-data[column].min() + 1)

        if method == 'log':
            data[column] = np.log(data[column])
        elif method == 'sqrt':
            data[column] = np.sqrt(data[column])
        elif method == 'box-cox':
            # Apply Box-Cox transformation only if all values are positive
            if data[column].min() > 0:
                data[column], _ = stats.boxcox(data[column])
            else:
                # Using a generalized Yeo-Johnson transformation 
                # if Box-Cox is not possible
                pt = PowerTransformer(method='yeo-johnson')
                data[column] = pt.fit_transform(data[[column]]).flatten()

    return data

def validate_skew_method(data: Series, method: str):
    """
    Validates the appropriateness of a skewness correction method based on the
    characteristics of the data provided. It ensures that the chosen method can
    be applied given the nature of the data's distribution, such as the presence
    of non-positive values which may affect certain transformations.

    Parameters
    ----------
    data : pandas.Series
        A Series containing the data to be checked for skewness correction.
    method : str
        The method of transformation intended to correct skewness:
        - 'log' : Natural logarithm, requires all positive values.
        - 'sqrt': Square root, requires non-negative values.
        - 'box-cox': Box-Cox transformation, requires all positive values.
          Falls back to Yeo-Johnson if non-positive values are found.

    Raises
    ------
    ValueError
        If the selected method is not suitable for the data based on its values.

    Returns
    -------
    str
        A message confirming the method's suitability or suggesting an
        alternative.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.dataops import validate_skew_method
    >>> data = pd.Series([0.1, 1.5, 3.0, 4.5, 10.0])
    >>> print(validate_skew_method(data, 'log'))
    >>> data_with_zeros = pd.Series([0, 1, 2, 3, 4])
    >>> print(validate_skew_method(data_with_zeros, 'log'))
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"Expected a pandas Series, but got"
                        f" {type(data).__name__!r} instead.")

    min_value = data.min()
    if method == 'log':
        if min_value <= 0:
            raise ValueError(
                "Log transformation requires all data points to be positive. "
                "Consider using 'sqrt' or 'box-cox' method instead.")
    elif method == 'sqrt':
        if min_value < 0:
            raise ValueError(
                "Square root transformation requires all data points"
                " to be non-negative. Consider using 'box-cox' or a specific"
                " transformation that handles negative values.")
    elif method == 'box-cox':
        if min_value <= 0:
            return ("Box-Cox transformation requires positive values, but"
                    " non-positive values are present. Applying Yeo-Johnson"
                    " transformation instead as a fallback.")
    else:
        raise ValueError("Unsupported method provided. Choose"
                         " from 'log', 'sqrt', or 'box-cox'.")

    return f"The {method} transformation is appropriate for this data."

@isdf
def check_skew_methods_applicability(
        data: DataFrame) -> Dict[str, List[str]]:
    """
    Evaluates each numeric column in a DataFrame to determine which skew
    correction methods are applicable based on the data's characteristics. 
    It utilizes the `validate_skew_method` function to check the applicability
    of 'log', 'sqrt', and 'box-cox' transformations for each column.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame whose columns are to be evaluated for skew correction
        applicability.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary where keys are column names and values are lists of 
        applicable skew correction methods.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import check_skew_methods_applicability
    >>> df = pd.DataFrame({
    ...     'A': [0.1, 1.5, 3.0, 4.5, 10.0],
    ...     'B': [1, 2, 3, 4, 5],
    ...     'C': [-1, -2, -3, -4, -5],
    ...     'D': [0, 0, 1, 2, 3]
    ... })
    >>> applicable_methods = check_skew_methods_applicability(df)
    >>> print(applicable_methods)
    """
    applicable_methods = {}
    for column in data.select_dtypes(include=[np.number]):
        methods = []
        for method in ['log', 'sqrt', 'box-cox']:
            try:
                validate_skew_method(data[column], method)
                methods.append(method)
            except ValueError as e:
                 # Optionally log or handle this information
                print(f"Column '{column}': {str(e)}") 

        applicable_methods[column] = methods

    return applicable_methods

@Dataify(auto_columns=True)
def handle_duplicates(
    data: DataFrame, 
    return_duplicate_rows: bool=False, 
    return_indices: bool=False, 
    operation: str='drop'
    ):
    """
    Handles duplicate rows in a DataFrame based on user-specified options.
    
    This function can return a DataFrame containing duplicate rows, the indices
    of these rows, or remove duplicates based on the specified operation.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame in which to handle duplicates.
    return_duplicate_rows : bool, optional
        If True, returns a DataFrame containing all duplicate rows. 
        This parameter takes precedence over `return_indices` and `operation`
        if set to True.
        Defaults to False.
    return_indices : bool, optional
        If True, returns a list of indices of duplicate rows. This will only
        take effect if `return_duplicate_rows` is False and takes precedence
        over `operation`.
        Defaults to False.
    operation : {'drop', 'none'}, optional
        Specifies the operation to perform on duplicate rows:
        - 'drop': Removes all duplicate rows, keeping the first occurrence.
        - 'none': No operation on duplicates; the original DataFrame is returned.
        Defaults to 'drop'.

    Returns
    -------
    pandas.DataFrame or list
        Depending on the parameters provided, this function may return:
        - A DataFrame of duplicates if `return_duplicate_rows` is True.
        - A list of indices of duplicate rows if `return_indices` is True.
        - A DataFrame with duplicates removed if `operation` is 'drop'.
        - The original DataFrame if `operation` is 'none'.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.dataops import handle_duplicates
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 2, 4, 5, 1],
    ...     'B': [1, 2, 2, 4, 5, 1]
    ... })
    >>> print(handle_duplicates(data, return_duplicate_rows=True))
    >>> print(handle_duplicates(data, return_indices=True))
    >>> print(handle_duplicates(data, operation='drop'))

    Notes
    -----
    The function is designed to handle duplicates flexibly, allowing users to
    explore, identify, or clean duplicates based on their specific requirements. 
    It is useful in data cleaning processes where duplicates might skew the results
    or affect the performance of data analysis and predictive modeling.

    See Also
    --------
    pandas.DataFrame.duplicated : Check for duplicate rows in a DataFrame.
    pandas.DataFrame.drop_duplicates : Remove duplicate rows from a DataFrame.
    """
    operation = parameter_validator('operation', ["drop", "none"])(operation)
    # Identify all duplicates based on all columns
    duplicates = data.duplicated(keep=False)
    
    # Return DataFrame of duplicate rows if requested
    if return_duplicate_rows:
        return data[duplicates]

    # Return indices of duplicate rows if requested
    if return_indices:
        return data[duplicates].index.tolist()

    # Remove duplicate rows if operation is 'drop'
    if operation == 'drop':
        return data.drop_duplicates(keep='first')

    # Return the original DataFrame if no operation
    # is specified or understood
    return data

def handle_unique_identifiers(
    data: DataFrame,
    threshold: float = 0.95, 
    action: str = 'drop', 
    transform_func: Optional[Callable[[any], any]] = None
    ) -> pd.DataFrame:
    """
    Examines columns in the DataFrame and handles columns with a high proportion 
    of unique values. These columns can be either dropped or transformed based 
    on specified criteria, facilitating better data analysis and modeling 
    performance by reducing the number of effectively useless features.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to process for unique identifier columns.
    threshold : float, optional
        The proportion threshold above which a column is considered to have too 
        many unique values (default is 0.95). If the proportion of unique values 
        in a column exceeds this threshold, an action is taken based on the 
        'action' parameter.
    action : str, optional
        The action to perform on columns exceeding the unique value threshold:
        - 'drop': Removes the column from the DataFrame.
        - 'transform': Applies a function specified by 'transform_func' to the column.
        Default is 'drop'.
    transform_func : Callable[[any], any], optional
        A function to apply to columns where the 'action' is 'transform'. This function 
        should take a single value and return a transformed value. If 'action' is 
        'transform' and 'transform_func' is None, no transformation is applied.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with columns modified according to the specified action and 
        threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'ID': range(1000),
    ...     'Age': [25, 30, 35] * 333 + [40],
    ...     'Salary': [50000, 60000, 75000, 90000] * 250
    ... })
    >>> processed_data = handle_unique_identifiers(data, action='drop')
    >>> print(processed_data.columns)

    >>> def cap_values(val):
    ...     return min(val, 100)  # Cap values at 100
    >>> processed_data = handle_unique_identifiers(data, action='transform', 
    ...                                           transform_func=cap_values)
    >>> print(processed_data.head())

    Notes
    -----
    Handling columns with a high proportion of unique values is essential in data 
    preprocessing, especially when preparing data for machine learning models. 
    High-cardinality features may lead to overfitting and generally provide little 
    predictive power unless they can be meaningfully transformed.

    See Also
    --------
    pandas.DataFrame.nunique : Count distinct observations over requested axis.
    pandas.DataFrame.apply : Apply a function along an axis of the DataFrame.
    """
    action = parameter_validator('action', ["drop", "transform"])(action)
    # Iterate over columns in the DataFrame
    for column in data.columns:
        # Calculate the proportion of unique values
        unique_proportion = data[column].nunique() / len(data)

        # If the proportion of unique values is above the threshold
        if unique_proportion > threshold:
            if action == 'drop':
                # Drop the column from the DataFrame
                data = data.drop(column, axis=1)
            elif action == 'transform' and transform_func is not None:
                # Apply the transformation function if provided
                data[column] = data[column].apply(transform_func)

    # Return the modified DataFrame
    return data

if __name__ == "__main__":
    # Example usage of the function

    data_positive = pd.Series([0.1, 1.5, 3.0, 4.5, 10.0])
    try:
        print(validate_skew_method(data_positive, 'log'))
    except ValueError as e:
        print(e)

    data_with_negatives = pd.Series([-1, 2, 5, 7, 9])
    try:
        print(validate_skew_method(data_with_negatives, 'sqrt'))
    except ValueError as e:
        print(e)
            
    # Create a sample DataFrame
    data = pd.DataFrame({
        'ID': range(100),  # Unique identifier
        'Age': [20] * 25 + [30] * 25 + [40] * 25 + [50] * 25,
        'Salary': [50000 + x for x in range(100)],
    })

    # Define a simple transformation function to demonstrate the transform option
    def example_transform(x):
        return f"ID_{x}"

    # Handling unique identifiers by dropping them
    result_drop = handle_unique_identifiers(data, threshold=0.9, action='drop')
    print("DataFrame after dropping high-uniqueness columns:\n", result_drop.head())

    # Handling unique identifiers by transforming them
    result_transform = handle_unique_identifiers(
        data, threshold=0.9, action='transform', transform_func=example_transform)
    print("DataFrame after transforming high-uniqueness columns:\n", result_transform.head())

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame with duplicate rows
    data = pd.DataFrame({
        'A': [1, 2, 2, 4, 5, 1],
        'B': [1, 2, 2, 4, 5, 1],
        'C': [1, 2, 2, 4, 5, 1]
    })

    # Finding and returning duplicate rows
    duplicates_df = handle_duplicates(data, return_duplicate_rows=True)
    print("Duplicate Rows:\n", duplicates_df)

    # Returning the indices of duplicate rows
    duplicate_indices = handle_duplicates(data, return_indices=True)
    print("Indices of Duplicate Rows:\n", duplicate_indices)

    # Dropping duplicates and returning the cleaned DataFrame
    cleaned_data = handle_duplicates(data, operation='drop')
    print("Data without Duplicates:\n", cleaned_data)

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame with skew
    data = pd.DataFrame({
        'A': np.random.normal(0, 2, 100),
        'B': np.random.chisquare(2, 100),
        'C': np.random.beta(2, 5, 100) * 60,  # positive and skewed
        'D': np.random.uniform(-1, 0, 100)  # contains negative values
    })

    # Apply transformation
    result_log = handle_skew(data.copy(), method='log')
    result_sqrt = handle_skew(data.copy(), method='sqrt')
    result_boxcox = handle_skew(data.copy(), method='box-cox')

    print("Log-transformed:\n", result_log.head())
    print("Square root-transformed:\n", result_sqrt.head())
    print("Box-Cox transformed:\n", result_boxcox.head())


    # Create a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 3, 2, 1],
        'D': [2, 3, 2, 3, 2]
    })

    # Call the function to drop correlated features
    result = drop_correlated_features(data, threshold=0.8)
    print(result)


    # Create a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 3, 2, 1],
        'D': [2, 3, 2, 3, 2]
    })

    # Call the function
    result = correlation_ops(data, correlation_type='strong positive')
    print("Strong Positive Correlations:", result)

    import pandas as pd
    #from gofast.tools.dataops import data_assistant 
    # Create a sample DataFrame
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, None],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'City': ['New York', 'Los Angeles', 'San Francisco', 'Houston', 'Seattle']
    })
    # Call the assistant function
    data_assistant(df)









