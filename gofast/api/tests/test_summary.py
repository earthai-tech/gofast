# -*- coding: utf-8 -*-
import pytest 
import unittest
import numpy as np 
import pandas as pd
import  gofast.api as gf 
from gofast.api.summary import ReportFactory
from gofast.api.summary import Summary
from importlib import reload
import gofast.api.testing
reload(gofast.api.testing)

class TestReportFactory(unittest.TestCase):
    def setUp(self):
        # This method will be run before each test
        self.report_factory = ReportFactory(title="Test Report")

    def test_mixed_types_summary(self):
        report_data = {
            'Total Sales': 123456.789,
            'Average Rating': 4.321,
            'Number of Reviews': 987
        }
        expected_output = """
        ================================
                  Test Report           
        --------------------------------
        Total Sales       : 123456.7890
        Average Rating    : 4.3210
        Number of Reviews : 987
        ================================
        """
        gf.testing.assert_report_mixed_types(
            self.report_factory, report_data, expected_output)

    def test_recommendations(self):
        recommendations_text = "Consider increasing marketing budget."
        expected_output = (
            "========================\n"
            "       Test Report      \n"
            "------------------------\n"
            "Consider increasing marketing budget.\n"
            "========================"
        )
        gf.testing.assert_report_recommendations(
            self.report_factory, recommendations_text, expected_output)

    def test_model_performance_summary(self):
        model_results = {
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.92
        }
        expected_output = (
            "========================\n"
            "       Test Report      \n"
            "------------------------\n"
            "accuracy    : 0.95\n"
            "precision   : 0.93\n"
            "recall      : 0.92\n"
            "========================"
        )
        gf.testing.assert_report_model_performance(
            self.report_factory, model_results, expected_output)

    def test_data_summary(self):
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Salary': [50000, 60000, 70000, 80000, 90000]
        })
        # Assuming format_dataframe function provides a specific formatted string output
        expected_output = "Formatted DataFrame summary output"
        gf.testing.assert_report_data_summary(self.report_factory, df, expected_output)

# class TestSummary(unittest.TestCase):
#     def setUp(self):
#         # Setup runs before each test method
#         self.summary = Summary(title="Employee Data Overview")

#     def test_basic_statistics(self):
#         df = pd.DataFrame({
#             'Age': [25, 30, 35, 40, np.nan],
#             'Salary': [50000, 60000, 70000, 80000, 90000]
#         })
#         expected_output = "Expected formatted string of basic statistics"
#         self.summary.basic_statistics(df, include_correlation=True)
#         gf.testing.assert_summary_basic_statistics(self.summary, expected_output)

#     def test_unique_counts(self):
#         df = pd.DataFrame({
#             'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR'],
#             'Age': [28, 34, 45, 32, 41]
#         })
#         expected_output = "Expected formatted string of unique counts"
#         self.summary.unique_counts(df, include_sample=True, sample_size=3)
#         gf.testing.assert_summary_unique_counts(self.summary, expected_output)


# class TestSummary2(unittest.TestCase):
#     def setUp(self):
#         # Setup runs before each test method
#         self.data = {
#             'Age': [25, 30, 35, 40, np.nan],
#             'Salary': [50000, 60000, 70000, 80000, 90000],
#             'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR']
#         }
#         self.df = pd.DataFrame(self.data)
#         self.summary = Summary(title="Employee Data Overview")

#     def test_correlation_matrix_inclusion(self):
#         # Test whether the correlation matrix is correctly included
#         expected_output = "Expected formatted string including correlation matrix"
#         self.summary.basic_statistics(self.df, include_correlation=True)
#         gf.testing.assert_summary_correlation_matrix(self.summary, expected_output)

#     def test_data_sample_inclusion(self):
#         # Test whether the data sample is correctly included and formatted
#         expected_output = "Expected formatted string including a data sample"
#         self.summary.unique_counts(self.df, include_sample=True, sample_size=2)
#         gf.testing.assert_summary_data_sample(self.summary, expected_output, sample_size=2)

#     def test_summary_completeness(self):
#         # Test the completeness of the summary, combining various components
#         expected_output = "Expected complete summary output string"
#         self.summary.basic_statistics(self.df, include_correlation=True)
#         self.summary.unique_counts(self.df, include_sample=True, sample_size=2)
#         # ssert_summary_completeness is a comprehensive test function
#         gf.testing.assert_summary_completeness(self.summary, expected_output)

#     # Add tests for any edge cases or specific scenarios
#     def test_empty_dataframe(self):
#         # Test handling of empty DataFrame
#         empty_df = pd.DataFrame()
#         expected_output = "Expected output for an empty DataFrame"
#         self.summary.basic_statistics(empty_df)
#         # assert_summary_empty_dataframe is tailored for this scenario
#         gf.testing.assert_summary_empty_dataframe(self.summary, expected_output)

#     # Test for expected failures or warnings
#     def test_invalid_input(self):
#         # Test handling of invalid input (not a DataFrame)
#         invalid_data = "This is not a DataFrame"
#         with self.assertRaises(ValueError):
#             self.summary.basic_statistics(invalid_data)



if __name__ == '__main__':
    pytest.main([__file__])
