# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides functionalities for deploying and serving machine learning models,
including model inference, API integration, and real-time prediction 
capabilities.
"""

import os
import pickle
import joblib
import warnings
from typing import Tuple, List, Any, Dict, Optional, Union

from ...core.io import EnsureFileExists 

__all__=['fetch_model', 'load_model']

@EnsureFileExists
def fetch_model(
    file: str,
    path: Optional[str] = None,
    default: bool = False,
    name: Optional[str] = None,
    verbose: int = 0
    ) -> Union[Dict[str, Any], List[Tuple[Any, Dict[str, Any], Any]]]:
    """
    Fetches a model saved using the Python pickle module or joblib module.

    Parameters
    ----------
    file : str
        The filename of the dumped model, saved using `joblib` or Python
        `pickle` module.
    path : Optional[str], optional
        The directory path containing the model file. If None, `file` is assumed
        to be the full path to the file.
    default : bool, optional
        If True, returns a list of tuples (model, best parameters, best scores)
        for each model in the file. If False, returns the entire contents of the
        file.
    name : Optional[str], optional
        The name of the specific model to retrieve from the file. If specified,
        only the named model and its parameters are returned.
    verbose : int, optional
        Verbosity level. More messages are displayed for values greater than 0.

    Returns
    -------
    Union[Dict[str, Any], List[Tuple[Any, Dict[str, Any], Any]]]
        Depending on the `default` flag:
        - If `default` is True, returns a list of tuples containing the model,
          best parameters, and best scores for each model in the file.
        - If `default` is False, returns the entire contents of the file, which
          could include multiple models and their respective information.

    Raises
    ------
    FileNotFoundError
        If the specified model file is not found.
    KeyError
        If `name` is specified but not found in the loaded model file.

    Examples
    --------
    >>> model_info = fetch_model('model.pkl', path='/models',
                                 name='RandomForest', default=True)
    >>> model, best_params, best_scores = model_info[0]
    """
    full_path = os.path.join(path, file) if path else file

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File {full_path!r} not found.")

    is_joblib = full_path.endswith('.pkl') or full_path.endswith('.joblib')
    load_func = joblib.load if is_joblib else pickle.load
    with open(full_path, 'rb') as f:
        model_data = load_func(f)

    if verbose > 0:
        lib_used = "joblib" if is_joblib else "pickle"
        print(f"Model loaded from {full_path!r} using {lib_used}.")

    if name:
        try:
            specific_model_data = model_data[name]
        except KeyError:
            available_models = list(model_data.keys())
            raise KeyError(f"Model name '{name}' not found. Available models: {available_models}")
        
        if default:
            if not isinstance(specific_model_data, dict):
                warnings.warn(
                    "The retrieved model data does not follow the expected structure. "
                    "Each model should be represented as a dictionary, with the model's "
                    "name as the key and its details (including 'best_params_' and "
                    "'best_scores_') as nested dictionaries. For instance: "
                    "`model_data = {'ModelName': {'best_params_': <parameters>, "
                    "'best_scores_': <scores>}}`. As the structure is unexpected, "
                    "returning the raw model data instead of the processed tuple."
                )
                return specific_model_data
            # Assuming model data structure for specific named model when default is True
            return [(specific_model_data, specific_model_data.get('best_params_', {}),
                     specific_model_data.get('best_scores_', {}))]
        return specific_model_data

    if default:
        # Assuming model data structure contains 'best_model', 'best_params_', and 'best_scores'
        return [(model, info.get('best_params_', {}), info.get('best_scores_', {})) 
                for model, info in model_data.items()]

    return model_data

@EnsureFileExists
def load_model(
    file_path: str, *,
    retrieve_default: bool = True,
    model_name: Optional[str] = None,
    storage_format: Optional[str] = None
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Loads a saved model or data using Python's pickle or joblib module.

    Parameters
    ----------
    file_path : str
        The path to the saved model file. Supported formats are `.pkl` and `.joblib`.
    retrieve_default : bool, optional, default=True
        If True, returns the model along with its best parameters. If False,
        returns the entire contents of the saved file.
    model_name : Optional[str], optional
        The name of the specific model to retrieve from the saved file. If None,
        the entire file content is returned.
    storage_format : Optional[str], optional
        The format used for saving the file. If None, the format is inferred
        from the file extension. Supported formats are 'joblib' and 'pickle'.

    Returns
    -------
    Union[Any, Tuple[Any, Dict[str, Any]]]
        The loaded model or a tuple of the model and its parameters, depending
        on the `retrieve_default` value.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If the specified model name is not found in the file.
    ValueError
        If the storage format is not supported or if the loaded data is not
        a dictionary when a model name is specified.

    Example
    -------
    >>> model, params = load_model('path_to_file.pkl', model_name='SVC')
    >>> print(model)
    >>> print(params)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    storage_format = storage_format or os.path.splitext(file_path)[-1].lower().lstrip('.')
    if storage_format not in {"joblib", "pickle"}:
        raise ValueError(f"Unsupported storage format '{storage_format}'. "
                         "Use 'joblib' or 'pickle'.")

    load_func = joblib.load if storage_format == 'joblib' else pickle.load
    with open(file_path, 'rb') as file:
        loaded_data = load_func(file)

    if model_name:
        if not isinstance(loaded_data, dict):
            warnings.warn(
                f"Expected loaded data to be a dictionary for model name retrieval. "
               f"Received type '{type(loaded_data).__name__}'. Returning loaded data.")
            return loaded_data

        model_info = loaded_data.get(model_name)
        if model_info is None:
            available = ', '.join(loaded_data.keys())
            raise KeyError(f"Model '{model_name}' not found. Available models: {available}")

        if retrieve_default:
            if not isinstance(model_info, dict):
                # Check if 'best_model_' and 'best_params_' are among the keys
                main_keys = [key for key in loaded_data if key in (
                    'best_model_', 'best_params_')]
                if len(main_keys) == 0:
                    warnings.warn(
                    "The structure of the default model data is not correctly "
                    "formatted. Expected 'best_model_' and 'best_params_' to be "
                    "present within a dictionary keyed by the model's name. Each key "
                    "should map to a dictionary containing the model itself and its "
                    "parameters, for example: `{'ModelName': {'best_model_': <Model>, "
                    "'best_params_': <Parameters>}}`. Since the expected keys were "
                    "not found, returning the unprocessed model data."
                    )
                    return model_info
                else:
                    # Extract 'best_model_' and 'best_params_' from loaded_data
                    best_model = loaded_data.get('best_model_', None)
                    best_params = loaded_data.get('best_params_', {})
            else:
                # Direct extraction from model_info if it's properly structured
                best_model = model_info.get('best_model_', None)
                best_params = model_info.get('best_params_', {})

            return best_model, best_params

        return model_info

    return loaded_data