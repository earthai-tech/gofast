# -*- coding: utf-8 -*-
"""
test_stats_utils.py

@author: LKouadio <etanoyau@gmail.com>
"""

import pytest 

import numpy as np
import pandas as pd
from gofast.stats.utils import gomean, gomedian, gomode, gostd, govar
from gofast.stats.utils import get_range, quartiles

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

# # Happy path test cases
# def test_gomean_with_array():
#     print(gomean([1, 2, 3, 4, 5]))
#     assert gomean([1, 2, 3, 4, 5]) == 3.0

# def test_gomean_with_dataframe_all_columns(sample_dataframe):
#     result = gomean(sample_dataframe, as_frame=True)
#     assert all(result == pd.Series([3.0, 3.0], index=['A', 'B']))

# def test_gomean_with_dataframe_specific_columns(sample_dataframe):
#     assert gomean(sample_dataframe, columns=['A']) == 3.0

# # Visual check (might be skipped or modified for CI/CD)
# def test_gomean_visualization(sample_dataframe):
#     # This test might be more about ensuring no errors are raised
#     gomean(sample_dataframe, view=True, as_frame=True)

#     assert True  # Verify visualization code runs without error


# # Happy path test cases
# def test_gomedian_with_array():
#     assert gomedian([1, 2, 3, 4, 5]) == 3.0

# def test_gomedian_with_dataframe_all_columns(sample_dataframe):
#     result = gomedian(sample_dataframe, as_frame=True)
#     assert all(result == pd.Series([3.0, 3.0], index=['A', 'B']))

# def test_gomedian_with_dataframe_specific_columns(sample_dataframe):
#     assert gomedian(sample_dataframe, columns=['A']) == 3.0

# # Visual check
# def test_gomedian_visualization(sample_dataframe):
#     # This test ensures no errors are raised during visualization
#     gomedian(sample_dataframe, view=True, as_frame=True)
#     assert True  # Verify visualization code runs without error

# # Happy path test cases
# def test_gomode_with_array():
#     assert gomode([1, 2, 2, 3, 3, 3]) == 3

# def test_gomode_with_dataframe_all_columns(sample_dataframe):
#      result = gomode(sample_dataframe, as_frame=True)
#      # Assuming mode returns the first mode encountered
#      expected = pd.Series([1, 4], index=['A', 'B']) 
#      assert all(result == expected)
     
# @pytest.mark.skip 
# def test_gomode_with_dataframe_specific_columns(sample_dataframe):
#     result = gomode(sample_dataframe, columns=['B'], as_frame=True)
#     assert result.tolist() == [4, 5] 

# # # Visual check
# def test_gomode_visualization(sample_dataframe):
#     # This test ensures no errors are raised during visualization
#     gomode(sample_dataframe, view=True, as_frame=True)
#     assert True  # Verify visualization code runs without error

# # Edge case test cases
# def test_gomode_empty_input():
#     with pytest.raises(ValueError):
#         gomode([])

# def test_govar_with_array():
#     data_array = np.array([1, 2, 3, 4, 5])
#     expected_variance = 2.0 # calcualted the population variance 
#     assert govar(data_array, ddof=0) == expected_variance, "Variance calculation for numpy array failed"

# def test_govar_with_list():
#     data_list = [1, 2, 3, 4, 5]
#     expected_variance = 2.5 # sample variance is computed instead by default ddof=1
#     assert govar(data_list) == expected_variance, "Variance calculation for list failed"

# def test_govar_with_dataframe():
#     data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
#     expected_variance = pd.Series([1.0, 1.0], index=['A', 'B'])
#     pd.testing.assert_series_equal(
#         govar(data_df, as_frame=True), expected_variance, check_names=False)

# def test_govar_with_dataframe_columns_specified():
#     data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
#     expected_variance = 1.0
#     assert govar(data_df, columns=['A'], ddof=1) == expected_variance

# def test_govar_with_axis_specified():
#     data_array = np.array([[1, 2, 3], [4, 5, 6]])
#     expected_variance = np.array([2.25, 2.25, 2.25])
#     np.testing.assert_array_almost_equal(
#         govar(data_array, axis=0, ddof =0), expected_variance)

# @pytest.mark.skipif('not hasattr(govar, "view")')
# def test_govar_with_view_enabled():
#     data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
#     # This test assumes that visualization does not affect the return value
#     # and primarily ensures that no exceptions are raised during its execution.
#     try:
#         govar(data_df, view=True, fig_size=(8, 4))
#     except Exception as e:
#         pytest.fail(f"Visualization raised an exception: {e}")

# Test data
data_list = [1, 2, 3, 4, 5]
data_array = np.array([1, 2, 3, 4, 5])
data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# @pytest.mark.parametrize("input_data,expected_output", [
#     (data_list, np.std(data_list, ddof=1)),
#     (data_array, np.std(data_array, ddof=1)),
#     (data_df, pd.Series([1.0, 1.0], index=['A', 'B'])),
# ])
# def test_gostd_basic(input_data, expected_output):
#     # Test without additional parameters
#     result = gostd(input_data)
#     if isinstance(input_data, pd.DataFrame):
#         pd.testing.assert_series_equal(result, expected_output)
#     else:
#         assert result == pytest.approx(expected_output)

# def test_gostd_with_columns():
#     # Test specifying columns in a DataFrame
#     result = gostd(data_df, columns=['A'], ddof=1)
#     assert result == pytest.approx(1.0)

# @pytest.mark.parametrize("ddof,expected_output", [
#     (0, np.std(data_list, ddof=0)), (1, np.std(data_list, ddof=1))])
# def test_gostd_ddof(ddof, expected_output):
#     # Test with different ddof values
#     result = gostd(data_list, ddof=ddof)
#     assert result == pytest.approx(expected_output)

# def test_gostd_as_frame():
#     # Test returning result as a DataFrame
#     result = gostd(data_list, as_frame=True)
#     expected_output = pd.Series(np.std(data_list, ddof=1), index=['Standard Deviation'])
#     pd.testing.assert_series_equal(result, expected_output)

# def test_gostd_view(capfd):
#     # Test visualization (view=True) does not raise errors
#     # Note: This does not assert the correctness of the plot, only that the function runs
#     gostd(data_list, view=True)
#     out, err = capfd.readouterr()
#     assert err == ""

# @pytest.mark.parametrize("input_data, expected_exception", [
#     ('invalid_data_type', ValueError), 
#     # Non-numeric DataFrame,  Did not raise any error at least verbose is set. 
#   #   (pd.DataFrame({'A': ['a', 'b', 'c']}), ValueError),  
# ])
# def test_gostd_errors(input_data, expected_exception):
#     # Test error handling for invalid inputs
#     with pytest.raises(expected_exception):
#         gostd(input_data)

# Test data

data_df = pd.DataFrame({'A': [2, 5, 8], 'B': [1, 4, 7]})

# @pytest.mark.parametrize("input_data,expected_result", [
#     (data_array, pd.Series([4.0], index=['Range Values'])),  # Testing with a numpy array
#     (data_df, pd.Series([6.0, 6.0], index=['A', 'B'])),  # Testing with a DataFrame, default columns
# ])
# def test_get_range_basic(input_data, expected_result):
#     result = get_range(input_data, as_frame=True)
#     if isinstance(result, pd.Series):
#         pd.testing.assert_series_equal(result, expected_result)
#     else:
#         assert result == expected_result

# def test_get_range_with_columns():
#     # Testing with specified columns in a DataFrame
#     result = get_range(data_df, columns=['A'], as_frame=True)
#     expected_result = pd.Series([6.], index=['A'])
#     pd.testing.assert_series_equal(result, expected_result)

# @pytest.mark.parametrize("axis,expected_result", [
#     (0, pd.Series([6., 6.], index=['A', 'B'])),  # Testing across index (rows)
#     (1, pd.Series([1., 1., 1.], index=[0, 1, 2])),  # Testing across columns
# ])
# def test_get_range_axis(axis, expected_result):
#     result = get_range(data_df, axis=axis, as_frame=True)
#     pd.testing.assert_series_equal(result, expected_result)

# @pytest.mark.parametrize("input_data, expected_exception", [
#     ('invalid_data_type', ValueError), 
#     ])
# def test_get_range_errors(input_data, expected_exception):
#     # Testing with invalid input data type
#     with pytest.raises(expected_exception):
#         get_range(input_data)

# # Example of a test function that could be used to test view functionality,
# # though it's tricky to assert graphical output in a unit test.
# @pytest.mark.skip(reason="Difficult to test view functionality in unit tests")
# def test_get_range_view():
#     # This is a placeholder to indicate where a test for the view functionality might go
#     pass



@pytest.fixture
def sample_data():
    return np.array([1, 2, 3, 4, 5])

@pytest.fixture
def sample_dataframe2():
    return pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})

def test_quartiles_with_array(sample_data):
    result = quartiles(sample_data)
    expected = np.array([2., 3., 4.])
    np.testing.assert_array_equal(result, expected)

def test_quartiles_with_dataframe(sample_dataframe2):
    result = quartiles(sample_dataframe2, as_frame=True)
    expected = pd.DataFrame({'25%': [4.25, 3.25], '50%': [6., 5.], '75%': [7.25, 6.75]}, index=['A', 'B']).T
    pd.testing.assert_frame_equal(result, expected)

# def test_quartiles_with_specified_columns(sample_dataframe2):
#     result = quartiles(sample_dataframe2, columns=['A'], as_frame=True)
#     expected = pd.DataFrame({'25%': [4.25], '50%': [6.], '75%': [7.75]}, index=['A']).T
#     pd.testing.assert_frame_equal(result, expected)

# def test_quartiles_error_on_mismatched_lengths():
#     with pytest.raises(ValueError):
#         quartiles(np.array([1, 2, 3]), new_indexes=['a', 'b'])

# @pytest.mark.parametrize("plot_type", ['box', 'hist'])
# def test_quartiles_visualization_with_plot_types(sample_dataframe, plot_type):
#     # This test ensures no exceptions are raised during plotting
#     # Actual visualization output is not checked here
#     quartiles(sample_dataframe, view=True, plot_type=plot_type, fig_size=(8, 4))

# Additional tests can be added to cover other scenarios and parameter combinations


if __name__=="__main__": 
    # test_gomode_with_dataframe_specific_columns(sample_dataframe)
    pytest.main([__file__])



