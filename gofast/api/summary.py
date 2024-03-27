# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:29:22 2024

@author: Daniel
"""
import numpy as np
import pandas as pd

from gofast.api.formatter import DataFrameFormatter  # Assuming this is an available module

class ModelPerformanceReport:
    def __init__(self, model, X_test, y_test, metrics):
        """
        Initializes the report with model performance metrics.
        
        Parameters:
        - model: The trained model.
        - X_test: Test features DataFrame.
        - y_test: Test target variable.
        - metrics: List of metrics to calculate and display.
        """
        
    def display_summary(self):
        """Prints a summary of the model's performance."""
        
    def plot_metrics(self):
        """Generates plots for the specified metrics, such as ROC curves."""
        
    def detailed_report(self):
        """Generates a detailed text report on model's performance metrics."""
        
class DataFrameReport:
    def __init__(self, before_df, after_df, transformations):
        """
        Initializes the report with DataFrame transformations.
        
        Parameters:
        - before_df: DataFrame before transformations.
        - after_df: DataFrame after transformations.
        - transformations: List or dictionary of applied transformations.
        """
        
    def summary(self):
        """Displays summary statistics of the DataFrame before and after transformations."""
        
    def transformation_details(self):
        """Describes each transformation applied, including parameters and effects."""
        
class OptimizationReport:
    
    def __init__(self, optimization_result):
        """
        Initializes the report with optimization results.
        
        Parameters:
        - optimization_result: Object containing results from optimization.
        """
        
    def best_params(self):
        """Displays the best parameters found."""
        
    def performance_overview(self):
        """Displays performance metrics for the best model."""
        
    def convergence_plot(self):
        """Generates a plot showing the optimization process over iterations."""
        
class ReportFactory:
    @staticmethod
    def create_report(report_type, *args, **kwargs):
        """
        Factory method to create different types of reports.
        
        Parameters:
        - report_type: Type of the report to create (e.g., 'model_performance', 'dataframe', 'optimization').
        - args, kwargs: Arguments required to instantiate the report classes.
        """
        report_type = str(report_type).lower() 
        if report_type == 'model_performance':
            return ModelPerformanceReport(*args, **kwargs)
        elif report_type == 'dataframe':
            return DataFrameReport(*args, **kwargs)
        elif report_type == 'optimization':
            return OptimizationReport(*args, **kwargs)
        else:
            raise ValueError("Unknown report type")


class Report:
    def __init__(self, title=None, **kwargs):
        self.title = title
        self.report_data = kwargs

    def __str__(self):
       
        self.max_key_length = max((len(key) for key in self.report_data), default=0) + 4
        report_str = self.format_title(self.title, char='=') + "\n"
        dataframe_keys = []

        for key, value in self.report_data.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                dataframe_keys.append(key)
                continue
            report_str += self.format_item(key, value)
        
        for key in dataframe_keys:
            value = self.report_data[key]
            if isinstance(value, pd.Series):
                report_str += self.format_series(key, value, max_key_length)
            elif isinstance(value, pd.DataFrame):
                report_str += "\n" + f"{self.format_title(key, char='-'):^max_key_length}" + "\n"
                report_str += f"{DataFrameFormatter().add_df(value).__str__().replace ('=', '~'):^max_key_length}"
                report_str +="\n"

        report_str += "\n" + self.format_title(self.title, char='=') + "\n"
        return report_str
    

    def add_data_ops_report(self, report):
        self.report_data.update(report)

    @staticmethod
    def format_title(title, char='='):
        
        line_length = 100
        if title:
            return title.center(line_length, char)
        return char * smax_key_length

    @staticmethod
    def format_item(key, value, max_key_length):
        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        return f"{key.ljust(max_key_length)}: {value_str}\n"

    @staticmethod
    def format_series(key, series, max_key_length):
        series_str = f"{key} : Series: name={series.name} - values: {series.to_dict()}\n"
        return series_str

# class Report:
#     def __init__(self, title=None, **kwargs):
#         self.title = title
#         self.report_data = kwargs
#         self.subsection_styles = ['-', '~', '.', ':']  # Example subsection styles

#     def __str__(self):
#         report_str = self.format_title(self.title)
#         max_key_length = max(map(len, self.report_data.keys()), default=0)
        
#         for key, value in self.report_data.items():
#             if isinstance(value, pd.DataFrame):
#                 report_str += self.format_df_section(key, value, max_key_length)
#             elif isinstance(value, pd.Series):
#                 report_str += self.format_series(key, value, max_key_length)
#             elif isinstance(value, dict):
#                 report_str += self.format_nested_dict(key, value, 1, max_key_length)  # Level starts from 1
#             else:
#                 report_str += self.format_key_value(key, value, max_key_length)
        
#         report_str += self.format_title(self.title, char='=')  # Close the report with title underline
#         return report_str

#     def add_data_ops_report(self, report):
#         self.report_data.update(report)

#     def format_title(self, title, char='='):
#         if title:
#             return f"{title.center(80, char)}\n"
#         return "\n"

#     def format_key_value(self, key, value, max_key_length):
#         formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
#         return f"{key.ljust(max_key_length)} : {formatted_value}\n"

#     def format_series(self, key, series, max_key_length):
#         series_str = f"{key} : Series: name={series.name} - values: {series.to_dict()}\n"
#         return series_str

#     def format_df_section(self, key, df, max_key_length):
#         # Placeholder for dataframe section formatting
#         df_str = f"\n{'':-^{max_key_length}}\n"  # Subsection line
#         df_str += f"{key.center(max_key_length)}\n"
#         df_str += f"{'':-^{max_key_length}}\n"
#         # Assuming this is a static method returning string
#         df_str += DataFrameFormatter().add_df(df).__str__().replace ("=", '-')  
#         return df_str

#     def format_nested_dict(self, parent_key, nested_dict, level, max_key_length):
#         # Placeholder for nested dict formatting
#         dict_str = f"\n{parent_key.center(max_key_length)}\n"
#         dict_str += f"{self.subsection_styles[level % len(self.subsection_styles)] * max_key_length}\n"
#         for key, value in nested_dict.items():
#             nested_key = f"{parent_key}.{key}"
#             if isinstance(value, dict):
#                 dict_str += self.format_nested_dict(nested_key, value, level + 1, max_key_length)
#             else:
#                 dict_str += self.format_key_value(nested_key, value, max_key_length)
#         return dict_str

#     # Placeholder for DataFrameFormatter class method
#     def format_df(self, df):
#         return DataFrameFormatter().add_df(df)

# report = Report(title="Example Report")
# report.add_data_ops_report({
#     "key1": np.random.random(),
#     "key2": "value2",
#     "key3": pd.Series([1, 2, 3], name="series_name"),
#     "key4": pd.DataFrame(np.random.rand(4, 3), columns=["col1", "col2", "col3"])
# })
# print(report)
# =================================Example Report=================================
# key1 : 0.8391
# key2 : value2
# key3 : Series: name=series_name - values: {0: 1, 1: 2, 2: 3}

# ----
# key4
# ----
# -------------------------------------------------------------------
#                  col1                   col2                   col3
# -------------------------------------------------------------------
#                0.1158                 0.9090                 0.1917
#                0.9667                 0.3779                 0.4441
#                0.5020                 0.1207                 0.2004
#                0.6772                 0.0511                 0.4482
# -------------------------------------------------------------------=================================Example Report=================================

# I expected this 


# =================================Example Report=================================
# key1 : 0.8391
# key2 : value2
# key3 : Series: name=series_name - values: {0: 1, 1: 2, 2: 3}

#                                 key4
#            ---------------------------------------------------------
#                  col1                   col2                   col3
#            ---------------------------------------------------------
#                0.1158                 0.9090                 0.1917
#                0.9667                 0.3779                 0.4441
#                0.5020                 0.1207                 0.2004
#                0.6772                 0.0511                 0.4482
#            ---------------------------------------------------------
# ================================================================================
  
# class Report: 
    
#     def __init__ (self, title =None, **kwargs): 
#         self.title =title 
        
#     def __str__(self, ):
 
         
#         # report must be a dictionnary of key string and values  and the formatage 
#         # should be like this . where 'Report tile' is placed in center of the 
#         # line. If there is not title, line must continue 
#         # and key space is determine by the max length of all dictionnary keys. 
#         # and based on that the  ":" is placed. 
#         # Also the length of title line '=' and subsection line '-' or '~' or '.' etc 
#         # is selected based on the  max lenght of keys and formed values. 
#         # here is an example: 
            
#         # ============================ Report =================================
#         # key1                  :  value1 round(4 decimal) if numeric. 
#         # key2                  :  value2 
#         # key3                  :  value3 
#         # ...                   :  ... 
#         # key n                 :  value n 
#         # =====================================================================
        
#         # if value of a key is a series then convert the values to dict for the 
#         # series formatage.  For instance the dict key (e.g keyseries)  that has series 
#         # as value is named keyseries. if should be formated as : 
            
#         # ============================ Report =================================================
#         # key1                  :  value1 
#         # key2                  :  value2 
#         # key3                  :  value3 
#         # keyseries             : Series: name=<series_name> - values: < series value in dict> 
#         # ======================================================================================
  
#         # if value of key is pandas dataframe then draw a line '-' as subsection after 
#         # one blank line and  create a section like :
            
#         # for instance the dict key (e.g keydf)  that has value ('valuedf) should be 
#         # formated by appendding  the dataframe format make from the method 
#         # self.format_df  and and center key. The total format should be for instance: 
            
#         # ============================ Report =================================
#         # key1                  :  value1 
#         # key2                  :  value2 
#         # key3                  :  value3 
#         # ---------------------------------------------------------------------
#         #                       keydf [center] 
#         # ---------------------------------------------------------------------
#         # Append DataFrame format here from self.format_df. Since it is a subsection then replace 
#         # the "=" by "~" and center the formatage 
#         # Note that self.format use the class DataFrameFormater to produce the table. 
#         # if the length of table of dataframe formater is larger than the report tile length 
#         # adjust all line to fit the max lengh of all tables. 
        
#         # if the value of the key is a dict with nested dict, then format accordingly like the report 
#         # but change the  subsection every time. 
#         # Note that subsection line must be different everytimes, 
#         # like subsection 1: use '-' 
#         # like subsection 2: use '~' 
#         # like subsection 3: use < selected accordingly> 
#         # ...
        
#         # Note the level of each subsubsection should be a join of 
#         # subsecction key and current section. For instance  a dict like {key_level1: { keylevel2: ...}}
#         # the subsecction keylevel2 shouldbe : key_level1.keylevel2 and so on 
        
#         # Here is a gobal examples of  a report that contain key as str, series, datarame and nested dict 
#         # should present like 
        
#         # ============================   Report  ===============================================
#         # key1                  :  value1 round(4 decimal) if numeric. 
#         # key2                  :  value2 
#         # key3                  :  value3 
#         # ---------------------------------------------------------------------------------------
#         # [ Leave one space of entire line ]
#         #                       key4 [ this ket has value in dataframe] [ Note that this table is yield from DataFrameFormater ]
#         #            ------------------------------------------------------
#         #            column 1          column2           column3        ...
#         #            valuecol1         valuecol2         valuecol3      ...
#         #            ...               ...               ...            ...
#         #            ------------------------------------------------------
        
#         # [ Leave one space of entire line ]
#         # key                   : Series: name=<series_name> - values: < series value in dict> 
#         # key6                  : value6
#         # ---------------------------------------------------------------------------------------
#         #                      
#         #                     key7 [this key contain nested dict]
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # key7.key_nested1             : value_nested1 
#         # key7.key_nested2             : value_nested2 etc ... 
        
#         # =========================================================================================
        
#         # finally close the report line with '=' 
        
#         # skip the documentation for brievity and use the best Python skill for 
#         # summary report making. Elevate the Python programming skill. 
        
#     def add_data_ops_report( self, report ): 
        
#     def format_df(self, df ): 
        
#         from gofast.api.formatter import DataFrameFormatter 
        
#         # DataFrameFormatter  yield a table like this: 
            
#         # ==============================
#         #         column1    column2    
#         # ------------------------------
#         # Index1   1.0000     5.6789     
#         # Index2   2.0000     6.0000     
#         # Index3   3.1235     7.0000     
#         # Index4   4.0000     8.0000     
#         # ==============================

    
#     def format_nested_dict (self, dict_): 
        
        
    
    
        
        
        
        
        
        
        
        

        


    

#XXX TODO 
class CustomDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def summary(self, include_correlation=False, include_uniques=False, 
                statistics=['mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                include_sample=False, sample_size=5):
        summary = {
            "Shape": self.shape,
            "Data Types": self.dtypes.to_dict(),
            "Missing Values": self.isnull().sum().to_dict(),
            "Basic Statistics": {}
        }
        
        # Basic statistics for numeric columns based on user-selected statistics
        if statistics:
            summary["Basic Statistics"] = self.describe().loc[statistics].to_dict()
        
        # Correlation matrix for numeric columns
        if include_correlation and self.select_dtypes(include=['number']).shape[1] > 1:
            summary["Correlation Matrix"] = self.corr().round(2).to_dict()
        
        # Unique counts for categorical columns
        if include_uniques:
            cat_cols = self.select_dtypes(include=['object', 'category']).columns
            summary["Unique Counts"] = {col: self[col].nunique() for col in cat_cols}
        
        # Sample of the data
        if include_sample:
            if sample_size > len(self):
                sample_size = len(self)
            summary["Sample Data"] = self.sample(n=sample_size).to_dict(orient='list')
        
        return summary




# class AnovaResults:
#     """
#     Anova results class

#     Attributes
#     ----------
#     anova_table : DataFrame
#     """
#     def __init__(self, anova_table):
#         self.anova_table = anova_table

#     def __str__(self):
#         return self.summary().__str__()

#     def summary(self):
#         """create summary results

#         Returns
#         -------
#         summary : summary2.Summary instance
#         """
#         summ = summary2.Summary()
#         summ.add_title('Anova')
#         summ.add_df(self.anova_table)

#         return summ
    
class Summary:
    def __init__(self, data):
        """
        Initialize with either a pandas DataFrame or a 'bunch' object (similar to sklearn's Bunch).
        The object should contain either the data for summary1 or model results for summary2.
        """
        self.data = data

    def summary1(self):
        """
        Display basic statistics of a DataFrame, with numerical values rounded 
        to 4 decimal places.
        """
        if isinstance(self.data, pd.DataFrame):
            # Basic info
            print("                               Data                               ")
            print("="*75)
            print("                               Core                               ")
            print("-"*75)
            print("No rows      No columns    Types        NaN exists?    %NaN (axis=0:axis1)")    
            print(f"{self.data.shape[0]:<12} {self.data.shape[1]:<13} {self.data.dtypes.nunique():<11} {self.data.isnull().any().any()}")              
            print("-"*75)
            print("                            Statistics                            ")
            print("-"*75)
            
            # Using pandas describe to get summary statistics and rounding
            desc = self.data.describe(include='all').applymap(
                lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
            for feature in desc.columns:
                stats = desc[feature]
                print(f"{feature:<12}", end="")
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    print(f"{stats[stat]:>15}", end="")
                print()
            print("="*75)
        else:
            print("Data is not a DataFrame. Please provide a DataFrame for summary1.")


    def summary2(self):
        """
        Display fine-tuned model results.
        Assuming 'data' is a 'bunch' object containing model results and statistics.
        """
        if hasattr(self.data, 'estimator') and hasattr(self.data, 'cv_results'):
            print("                              Results                              ")
            print("="*75)
            print("                               Main                                ")
            print("-"*75)
            print(f"Estimator         : {self.data.estimator}")
            print(f"Best parameters   : {self.data.best_params}")
            print(f"nCV               : {self.data.cv}")
            print(f"Scoring           : {self.data.scoring}")
            print("-"*75)
            print("                             CV Results                            ")
            print("-"*75)
            
            cv_results = self.data.cv_results
            for i in range(self.data.cv):
                print(f"Cv{i+1:<5} {cv_results['mean_test_score'][i]:<15} {cv_results['split{i}_test_score']:<12} {cv_results['std_test_score'][i]:<15} {np.nanmean(cv_results['mean_test_score']):<12}")
            print("="*75)
        else:
            print("Data does not contain model results. Please provide a 'bunch' object for summary2.")

# Example usage
# You would replace these example calls with actual data or model results
# summary_instance = Summary(your_dataframe_or_bunch_object)
# summary_instance.summary1()  # For DataFrame statistics
# summary_instance.summary2()  # For model results


    """ Gofast Summary class must micmic the statmodels display models. 
    Gofast object should be encapsulated in bunch object wher each attributes 
    can be displayed. 
    Three method of display should be created. 
    
    summary1 (self , etc...) should reflect to display the minor 
    staticstic of any dataframe. for instance 
    - df.summary1() should display as statmodels display the 
    Number features, number of row, datatypes and statistic likes 
    missing values,  ['mean', 'std', 'min', '25%', '50%', '75%', 'max'] etc. 
    
                               Data 
    ===========================================================================
                               Core
    ---------------------------------------------------------------------------
    No rows      No columns    No num-feat.   No cat-feat    %NaN [ axis0 |axis1]
    ******************************************************************************
    123           7             3              4               
    -----------------------------------------------------------------------------
                               Statistics 
    ---------------------------------------------------------------------------
    mean          std         min        max         25%         50%      75%  
    ***************************************************************************
    Feature 1    ...          ....       ...       ....        ...       ....
    Feature 2    ...          ...        ...       ...         ...       ...
    ....
    
    ===========================================================================
    
    if NaN exists in dataframe , NaN exist? should be True, then compute the 
    % of NaN in the whole data and according to each axis.  For instance 
    %NaN [ axis0 |axis1] should display:   15[30:7] where 15 means 15% of Nan 
    in the whole data and 30% NaN found in axis 1 and 15% in axis 1
    
    also note that the numeric values of 
    
    
    - summary2() refers to display the fine-tuned models results. 
    if the bunch model is print then a nice table that micmics the statmodels 
    must be display that includes, estimator name, best parameters, ncv , 
    scoring and job specified . 
    then the cv results like : 
        
                                 Results 
    ===========================================================================
                                  Main 
    ----------------------------------------------------------------------------
    Estimator         : SVC 
    Best parameters   : .e.g {C: 1, gamma=0.1}
    nCV               :  e.g 4 
    scoring           : e.g accuracy
    ----------------------------------------------------------------------------
                                 CV results 
    ---------------------------------------------------------------------------
    Fold        Mean score        CV score      std score          Global mean 
    ***************************************************************************       
    Cv1          0.6789              0.6710       0.0458               0.7090
    cv2          0.5678              0.7806       0.8907               0.9089
    cv3          0.9807              0.6748       0.8990               0.7676
    cv4          0.8541              0.8967       0.9087               0.6780
    ===========================================================================
    
    The mean score can be computed for each fold of the cv results 
    same for cv scores, std score and global mean who exclude Nan if exists. 
    
    
    
    
    write the methods summary 1 and summary 2 to perform this task flexibility 
    and robust.skip the documentation for now 
    
    """
# Example usage
if __name__ == "__main__":
    report = Report(title="Example Report")
    report.add_data_ops_report({
        "key1": np.random.random(),
        "key2": "value2",
        "key3": pd.Series([1, 2, 3], name="series_name"),
        "key4": pd.DataFrame(np.random.rand(4, 3), columns=["col1", "col2", "col3"])
    })
    print(report)
    
    # Creating an example DataFrame
    df_data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, None, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux'],
        'D': [0.1, 0.2, 0.3, np.nan, 0.5]
    }
    df = CustomDataFrame(df_data)
    
    # Customizing the summary
    summary_ = df.summary(include_correlation=True, include_uniques=True, 
                         statistics=['mean', '50%', 'max'], include_sample=True
                         )
    for key, value in summary_.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")