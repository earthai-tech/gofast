# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 00:30:04 2024

@author: Daniel
"""
import pytest 
from unittest.mock import patch
import numpy as np
import pandas as pd
from gofast.datasets.load import load_hydro_metrics, load_statlog
from gofast.datasets.load import  load_dyspnea, load_hlogs, load_nlogs 
from gofast.datasets.load import load_bagoue

# Assuming these are the correct paths for the imports
# Adjust according to your project structure

# @patch("gofast.datasets.load._finalize_return")
# @patch("gofast.datasets.load._handle_split_X_y")
# @patch("gofast.datasets.load._prepare_data")
# @patch("gofast.datasets.load._load_data")
# @patch("gofast.datasets.load._ensure_data_file")
# @patch("gofast.datasets.load._setup_key")
# def test_load_hlogs_default(mock_setup_key, mock_ensure_data_file, mock_load_data, mock_prepare_data, mock_handle_split_X_y, mock_finalize_return):
#     mock_setup_key.return_value = ('h502', {"h502", "h2601"})
#     mock_load_data.return_value = pd.DataFrame({'feature1': [1, 2], 'remark': ['note1', 'note2']})
#     mock_prepare_data.return_value = (pd.DataFrame({'feature1': [1, 2]}), pd.DataFrame({'feature1': [1, 2]}), pd.Series([3, 4]), ['feature1'])
#     mock_handle_split_X_y.return_value = (pd.DataFrame({'feature1': [1]}), pd.DataFrame({'feature1': [2]}), pd.Series([3]), pd.Series([4]))
#     mock_finalize_return.return_value = "Bunch object"
    
#     # Default behavior without any flags
#     result = load_hlogs()
#     assert result == "Bunch object"
#     mock_setup_key.assert_called_once()
#     mock_ensure_data_file.assert_called_once()
#     mock_load_data.assert_called_once()
#     mock_prepare_data.assert_called_once()
#     mock_finalize_return.assert_called_once()
    
# def test_load_hlogs():
#     # Test with split_X_y=True and as_frame=True
#     X, Xt, Y, Yt = load_hlogs(split_X_y=True, as_frame=True)
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(Xt, pd.DataFrame)
#     assert isinstance(Y, pd.DataFrame)
#     assert isinstance(Yt, pd.DataFrame)


@patch("gofast.datasets.load.csv_data_loader")
@patch("gofast.datasets.load._to_dataframe")
def test_load_hydro_metrics_default(mock_to_dataframe, mock_csv_data_loader):
    # Setup mock return values
    mock_csv_data_loader.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        np.array([5, 6]),
        ['flow'],
        ['feature1', 'feature2'],
        "Hydro-Meteorological Dataset Description"
    )
    mock_to_dataframe.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]}),
        pd.Series([5, 6])
    )

    # Test return_X_y=False and as_frame=False (default behavior)
    bunch = load_hydro_metrics()
    assert 'data' in bunch and 'target' in bunch and 'DESCR' in bunch
    assert isinstance(bunch.data, np.ndarray)
    assert isinstance(bunch.target, np.ndarray)

@patch("gofast.datasets.load.csv_data_loader")
@patch("gofast.datasets.load._to_dataframe")
def test_load_hydro_metrics_as_frame(mock_to_dataframe, mock_csv_data_loader):
    # Setup mock return values
    mock_csv_data_loader.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        np.array([5, 6]),
        ['flow'],
        ['feature1', 'feature2'],
        "Hydro-Meteorological Dataset Description"
    )
    mock_to_dataframe.return_value = (
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'flow': [5, 6]}),
        pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]}),
        pd.Series([5, 6])
    )

    # Test return_X_y=True and as_frame=True
    X, y = load_hydro_metrics(return_X_y=True, as_frame=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ['feature1', 'feature2']
    assert y.name == 'flow'

# def test_load_hydro_metrics():
#     # Test default behavior
#     data, target = load_hydro_metrics(return_X_y=True)
#     assert isinstance(data, np.ndarray)
#     assert isinstance(target, np.ndarray)
    
#     # Test as_frame=True
#     df, target_series = load_hydro_metrics(return_X_y=True, as_frame=True)
#     assert isinstance(df, pd.DataFrame)
#     assert isinstance(target_series, pd.Series)

# def test_load_statlog():
#     # Test return_X_y=True
#     data, target = load_statlog(return_X_y=True)
#     assert isinstance(data, np.ndarray)
#     assert isinstance(target, np.ndarray)
    
#     # Test as_frame=True
#     df, target_series = load_statlog(return_X_y=True, as_frame=True)
#     assert isinstance(df, pd.DataFrame)
#     assert isinstance(target_series, pd.Series)

# def test_load_dyspnea():
#     # Test return_X_y=True
#     data, target = load_dyspnea(return_X_y=True)
#     assert isinstance(data, np.ndarray)
#     assert isinstance(target, np.ndarray)
    
#     # Test as_frame=True and split_X_y=True
#     X, Xt, y, yt = load_dyspnea(as_frame=True, split_X_y=True, test_size=0.3)
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(Xt, pd.DataFrame)
#     assert isinstance(y, pd.Series)
#     assert isinstance(yt, pd.Series)

# def test_load_nlogs():
#     # Test with as_frame=True
#     data, target = load_nlogs(return_X_y=True, as_frame=True)
#     assert isinstance(data, pd.DataFrame)
#     assert isinstance(target, pd.Series)
    
#     # Test with split_X_y=True and as_frame=True
#     X, Xt, y, yt = load_nlogs(split_X_y=True, as_frame=True)
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(Xt, pd.DataFrame)
#     assert isinstance(y, pd.Series)
#     assert isinstance(yt, pd.Series)

# def test_load_bagoue():
#     # Test return_X_y=True and as_frame=True
#     data, target = load_bagoue(return_X_y=True, as_frame=True)
#     assert isinstance(data, pd.DataFrame)
#     assert isinstance(target, pd.Series)
    
#     # Test split_X_y=True and as_frame=True
#     X, Xt, y, yt = load_bagoue(split_X_y=True, as_frame=True)
#     assert isinstance(X, pd.DataFrame)
#     assert isinstance(Xt, pd.DataFrame)
#     assert isinstance(y, pd.Series)
#     assert isinstance(yt, pd.Series)

if __name__ == "__main__":
    pytest.main([__file__])
