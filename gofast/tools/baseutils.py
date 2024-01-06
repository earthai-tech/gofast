# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
from __future__ import annotations, print_function 
import os
import h5py
import copy
import time
import shutil 
import pathlib
import warnings 
import threading
import subprocess
from six.moves import urllib 
from joblib import Parallel, delayed
import numpy as np 
import pandas as pd 
from tqdm import tqdm

from .._typing import Any,  List, NDArray, DataFrame, Optional
from .._typing import Dict, Union, TypeGuard, Tuple, ArrayLike
from ..exceptions import FileHandlingError 
from ..property import  Config
from .funcutils import is_iterable, ellipsis2false,smart_format,sPath 
from .funcutils import to_numeric_dtypes, assert_ratio 
from ._dependency import import_optional_dependency 
from .validator import array_to_frame, build_data_if 

def summarize_text_columns(
    data: DataFrame, /, column_names: List[str], 
    stop_words: str = 'english', encode: bool = False, 
    drop_original: bool = False, compression_method: Optional[str] = None
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
        'id': [1, 2],
        'column1': [
            "Sentence one. Sentence two. Sentence three.",
            "Another sentence one. Another sentence two. Another sentence three."
        ],
        'column2': [
            "More text here. Even more text here.",
            "Second example here. Another example here."
        ]
    }
    >>> df = pd.DataFrame(data)
    >>> summarized_df = summarize_text_columns(df, ['column1', 'column2'], 
                    stop_words='english', encode=True, drop_original=True, 
                    compression_method='mean')
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
        raise TypeError("The input data must be a pandas DataFrame.")
    
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
    >>> for msg in messages:
    ...     summary, encoding = simple_extractive_summary([msg], encode=True)
    ...     print(summary)
    ...     print(encoding) # encoding is the TF-IDF vector of the summary
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if len(texts) < 2:
        if not raise_exception:
            return (texts[0], None) if encode else texts[0]
        raise ValueError("The input list must contain at least two sentences for summarization.")

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
        augmented_df = augmented_df.append(synthetic_point, ignore_index=True)

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
      
      .. versionadded:: 0.2.5
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
       be used. 
      
       .. versionadded:: 0.2.7 
          Capable to read text and numpy formats ('.npy' and '.npz') data. Note
          that when data is stored in compressed ".npz" format, provided the 
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
    watex.tools.baseutils.save_or_load: 
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
    columns: List[str, ...]=None, 
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
   
def lowertify (*values, strip = True, return_origin: bool =... ): 
    """ Strip and convert value to lowercase. 
    
    :param value: str , value to convert 
    :return: value in lowercase and original value. 
    
    :Example: 
        >>> from gofast.tools.baseutils import lowertify 
        >>> lowertify ( 'KIND')
        Out[19]: ('kind',)
        >>> lowertify ( "KIND", return_origin =True )
        Out[20]: (('kind', 'KIND'),)
        >>> lowertify ( "args1", 120 , 'ArG3') 
        Out[21]: ('args1', '120', 'arg3')
        >>> lowertify ( "args1", 120 , 'ArG3', return_origin =True ) 
        Out[22]: (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
        >>> (kind, kind0) , ( task, task0 ) = lowertify(
            "KIND", "task ", return_origin =True )
        >>> kind, kind0, task, task0 
        Out[23]: ('kind', 'KIND', 'task', 'task ')
        """
    raw_values = copy.deepcopy(values ) 
    values = [ str(val).lower().strip() if strip else str(val).lower() 
              for val in values]

    return tuple (zip ( values, raw_values)) if ellipsis2false (
        return_origin)[0]  else tuple (values)

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
 
#XXX TODO      
def request_data (
    url:str, /, 
    task: str='get',
    data: Any=None, 
    as_json: bool=..., 
    as_text: bool = ..., 
    stream: bool=..., 
    raise_status: bool=..., 
    save2file: bool=..., 
    filename:str =None, 
    **kws
): 
    """ Fetch remotely data
 
    Request data remotely 
    https://docs.python-requests.org/en/latest/user/quickstart/#raw-response-content
    
    
    r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    r.status_code
    200
    r.headers['content-type']
    'application/json; charset=utf8'
    r.encoding
    'utf-8'
    r.text
    '{"type":"User"...'
    r.json()
    {'private_gists': 419, 'total_private_repos': 77, ...}
    
    """
    import_optional_dependency('requests' ) 
    import requests 
    
    as_text, as_json, stream, raise_status, save2file = ellipsis2false(
        as_text, as_json,  stream, raise_status , save2file)
    
    if task=='post': 
        r = requests.post(url, data =data , **kws)
    else: r = requests.get(url, stream = stream , **kws)
    
    if save2file and stream: 
        with open(filename, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    if raise_status: 
        r.raise_for_status() 
        
    return r.text if as_text else ( r.json () if as_json else r )

def get_remote_data(
    rfile:str, /,  
    savepath: str=None, 
    raise_exception: bool =True
): 
    """ Try to retrieve data from remote.
    
    Parameters 
    -------------
    rfile: str or PathLike-object 
       Full path to the remote file. It can be the path to the repository 
       root toward the file name. For instance, to retrieve the file 
       ``'AGSO.csv'`` which is located in ``gofast/etc/`` directory then the 
       full path should be ``'gofast/etc/AGSO.csv'``
        
    savepath: str, optional 
       Full path to place where to downloaded files should be located. 
       If ``None`` data is saved to the current directory.
     
    raise_exception: bool, default=True 
      raise exception if connection failed. 
      
    Returns 
    ----------
    status: bool, 
      ``False`` for failure and ``True`` otherwise i.e. successfully 
       downloaded. 
       
    """
    connect_reason ="""\
    ConnectionRefusedError: No connection could  be made because the target 
    machine actively refused it.There are some possible reasons for that:
     1. Server is not running as well. Hence it won't listen to that port. 
         If it's a service you may want to restart the service.
     2. Server is running but that port is blocked by Windows Firewall
         or other firewall. You can enable the program to go through 
         firewall in the inbound list.
    3. there is a security program on your PC, i.e a Internet Security 
        or Antivirus that blocks several ports on your PC.
    """  
    #git_repo , git_root= AGSO_PROPERTIES['GIT_REPO'], AGSO_PROPERTIES['GIT_ROOT']
    # usebar bar progression
    print(f"---> Please wait while fetching {rfile!r}...")
    try: import_optional_dependency ("tqdm")
    except:pbar = range(3) 
    else: 
        import tqdm  
        data =os.path.splitext( os.path.basename(rfile))[0]
        pbar = tqdm.tqdm (total=3, ascii=True, 
                          desc =f'get-{os.path.basename(rfile)}', 
                          ncols =97
                          )
    status=False
    root, rfile  = os.path.dirname(rfile), os.path.basename(rfile)
    for k in range(3):
        try :
            urllib.request.urlretrieve(root,  rfile )
        except: 
            try :
                with urllib.request.urlopen(root) as response:
                    with open( rfile,'wb') as out_file:
                        data = response.read() # a `bytes` object
                        out_file.write(data)
            except TimeoutError: 
                if k ==2: 
                    print("---> Established connection failed because"
                       "connected host has failed to respond.")
            except:pass 
        else : 
            status=True
            break
        try: pbar.update (k+1)
        except: pass 
    
    if status: 
        try: 
            pbar.update (3)
            pbar.close ()
        except:pass
        # print(f"\n---> Downloading {rfile!r} was successfully done.")
    else: 
        print(f"\n---> Failed to download {rfile!r}.")
    # now move the file to the right place and create path if dir not exists
    if savepath is not None: 
        if not os.path.isdir(savepath): 
            sPath (savepath)
        shutil.move(os.path.realpath(rfile), savepath )
        
    if not status:
        if raise_exception: 
            raise ConnectionRefusedError(connect_reason.replace (
                "ConnectionRefusedError:", "") )
        else: print(connect_reason )
    
    return status


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
    
    
def audit_data(
    data, 
    /, 
    dropna_threshold=0.5, 
    categorical_threshold=10, 
    standardize=True
    ):
    """
    Cleans and formats a dataset for analysis. 
    This includes handling missing values,
    converting data types, and standardizing numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to be cleaned and formatted.

    dropna_threshold : float, optional
        The threshold for dropping columns with missing values. 
        Columns with a fraction of missing values greater than 
        this threshold will be dropped.
        Default is 0.5 (50%).

    categorical_threshold : int, optional
        The maximum number of unique values in a column for it to 
        be considered categorical.
        Columns with unique values fewer than or equal to this 
        number will be converted to 
        categorical type. Default is 10.

    standardize : bool, optional
        If True, standardize numerical columns to have a mean of 0 
        and a standard deviation of 1.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The cleaned and formatted dataset.

    Example
    -------
    >>> data = pd.DataFrame({
    >>>     'A': [1, 2, np.nan, 4, 5],
    >>>     'B': ['x', 'y', 'z', 'x', 'y'],
    >>>     'C': [1, 2, 3, 4, 5]
    >>> })
    >>> clean_data = audit_data(data)

    """
    data = to_numeric_dtypes(data )
    dropna_threshold= assert_ratio(dropna_threshold)
    
    # Drop columns with too many missing values
    data = data.dropna(thresh=int(
        dropna_threshold * len(data)), axis=1)

    # Fill missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            # For categorical columns, fill with the mode
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # For numerical columns, fill with the median
            data[col] = data[col].fillna(data[col].median())

    # Convert columns to categorical if they have fewer
    # unique values than the threshold
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].nunique(
                ) <= categorical_threshold:
            data[col] = data[col].astype('category')

    # Standardize numerical columns
    if standardize:
        num_cols = data.select_dtypes(include=['number']).columns
        data[num_cols] = (
            data[num_cols] - data[num_cols].mean()) / data[num_cols].std()

    return data
    
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
    data : pd.DataFrame
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
    >>> run_with_progress_bar(["pip", "install", "some-package"])
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
    datasets: Optional[Dict[str, np.ndarray]] = None, 
    operation: str = 'store'
    ) -> Union[None, Dict[str, np.ndarray]]:
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
    datasets: Optional[Dict[str, pd.DataFrame]] = None, 
    operation: str = 'store') -> Union[None, Dict[str, pd.DataFrame]]:
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
    
def unified_storage(
    file_path: str,
    datasets: Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]] = None, 
    operation: str = 'store'
) -> Union[None, Dict[str, Union[np.ndarray, pd.DataFrame]]]:
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
        Required if operation is 'store'. Default is None.
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
    >>> df1 = pd.DataFrame(np.random.randint(0, 100, size=(200, 5)), columns=['A', 'B', 'C', 'D', 'E'])
    >>> handle_datasets_in_h5('my_datasets.h5', {'dataset1': data1, 'df1': df1}, operation='store')

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
    

    
    
    
    
    
    
    