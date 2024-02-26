# -*- coding: utf-8 -*-
"""
test_stats_utils.py

@author: LKouadio <etanoyau@gmail.com>
"""

import pytest 

import numpy as np
import pandas as pd
from gofast.stats.utils import gomean
from gofast.stats.utils import gomedian
from gofast.stats.utils import gomode

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

# Happy path test cases
# def test_gomean_with_array():
#     print(gomean([1, 2, 3, 4, 5]))
#     assert gomean([1, 2, 3, 4, 5]) == 3.0

# def test_gomean_with_dataframe_all_columns(sample_dataframe):
#     result = gomean(sample_dataframe, as_frame=True)
#     assert all(result == pd.Series([3.0, 3.0], index=['A', 'B']))

def test_gomean_with_dataframe_specific_columns(sample_dataframe):
    assert gomean(sample_dataframe, columns=['A']) == 3.0

# Visual check (might be skipped or modified for CI/CD)
# def test_gomean_visualization(sample_dataframe):
#     # This test might be more about ensuring no errors are raised
#     gomean(sample_dataframe, view=True, as_frame=True)

#     assert True  # Verify visualization code runs without error


# Happy path test cases
# def test_gomedian_with_array():
#     assert gomedian([1, 2, 3, 4, 5]) == 3.0

# def test_gomedian_with_dataframe_all_columns(sample_dataframe):
#     result = gomedian(sample_dataframe, as_frame=True)
#     assert all(result == pd.Series([3.0, 3.0], index=['A', 'B']))

# def test_gomedian_with_dataframe_specific_columns(sample_dataframe):
#     assert gomedian(sample_dataframe, columns=['A']) == 3.0

# Visual check
# def test_gomedian_visualization(sample_dataframe):
#     # This test ensures no errors are raised during visualization
#     gomedian(sample_dataframe, view=True, as_frame=True)
#     assert True  # Verify visualization code runs without error

# # Edge case test cases
# def test_gomedian_empty_input():
#     with pytest.raises(ValueError):
#         gomedian([])

# Happy path test cases
def test_gomode_with_array():
    assert gomode([1, 2, 2, 3, 3, 3]) == 3

def test_gomode_with_dataframe_all_columns(sample_dataframe):
     result = gomode(sample_dataframe, as_frame=True)
     # Assuming mode returns the first mode encountered
     expected = pd.Series([1, 4], index=['A', 'B']) 
     assert all(result == expected)

def test_gomode_with_dataframe_specific_columns(sample_dataframe):
    result = gomode(sample_dataframe, columns=['B'], as_frame=True)
    print("series 2 list =", result.tolist())
    assert result.tolist() == [4, 5] 

# # Visual check
# def test_gomode_visualization(sample_dataframe):
#     # This test ensures no errors are raised during visualization
#     gomode(sample_dataframe, view=True, as_frame=True)
#     assert True  # Verify visualization code runs without error

# # Edge case test cases
# def test_gomode_empty_input():
#     with pytest.raises(ValueError):
#         gomode([])


if __name__=="__main__": 
    
    pytest.main([__file__])



