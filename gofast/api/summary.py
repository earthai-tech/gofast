# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:29:22 2024

@author: Daniel
"""
import numpy as np
import pandas as pd

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





# class Report: 
    
#     def __init__ (self, title =None, **kwargs): 
#         self.title =title 
        
#     def __str__(self, ):
 
          # write a function to format a report 
#         # Report must be a dictionnary of key string and values 
          # where 'Report tile' is placed in center frame with a little space of the 
#         # line. If there is not title, line must continue 
#         # and key space is determine by the max length of all dictionnary keys 
#         # and ":" is placed accordingly. 
          # if the dictionnary value is numeric format accordingly and round 4 decimal. Here is example. 
#         # here is an example: 
            
#         # ======================= Report tile ===============================
#         # key1                  :  value1 round(4 decimal) if numeric. 
#         # key2                  :  value2 
#         # key3                  :  value3 
#         # ...                   :  ... 
#         # key n                 :  value n 
#         # ===================================================================
        
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
    
    
    - write a function  summary2(*args, **kwargs) refers to display the fine-tuned models results. 
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
# def format__inline_table ( 
#         contents, title =None, top_line ='=', sub_line ='-', bottom_line='=', 
#         table_length= None, header_content=None, max_table_length=None   ): 
#     # format_table_length 
    
#     if not isinstance ( content, dict): 
#         # raise informative error that inline table expect a dict of key and values 
#         # display like 
    #                              Results  [title center]
    # =========================================================================[top_line]
    #                               Main [ header content center]                                      
    # -------------------------------------------------------------------------[sub_line]
    # Estimator[key1]         : SVC [ value1]
    # Best parameters[key2]   : .e.g {C: 1, gamma=0.1}[value2]
    # nCV[key3]               :  e.g 4 [value3]
    # scoring[key4]           : e.g accuracy [value4]
    # ...                     : ...
    # -------------------------------------------------------------------------[ bottom_line]
    # the table line length is equal to sum of the [ maximum length of keys + space [' : '] + maximum length of value in  dicts. 
    # however if max table length is given check whether the maxkey length is less than the max_table_length, 
    # if max key is longer than max_table_length then ignore the max_table_length and use the default 
    # calculation as indicated above, however if the max_table_length is given and less than the 
    # max_table_length there compute the remained value length and if value length is higher than the 
    # remain length , use '...'  
    
    # if value in dict is numeric , format as 4 decimal. 
    # if header content is empty the subline must be removed as well as the header content not displayed
    # as 
    
    #                              Results  [title center]
    # =========================================================================[top_line]
    # Estimator[key1]         : SVC [ value1]
    # Best parameters[key2]   : .e.g {C: 1, gamma=0.1}[value2]
    # nCV[key3]               :  e.g 4 [value3]
    # scoring[key4]           : e.g accuracy [value4]
    # ...                     : ...
    # -------------------------------------------------------------------------[ bottom_line]

# Document the function below only. Use Numpy docstring and reST makup and come back 
# to new line after 65 characters. add examples and Notes. 

def summarize_table ( contents, 
    title=None, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='=', 
    ): 
    
    if not isinstance(contents, (pd.DataFrame, pd.Series)):
        raise ValueError("summarize_table expects a DataFrame of keys and values.")
    if isinstance (contents, pd.Series): 
        contents= contents.to_frame() 
       
        # Initial calculations for formatting
        max_key_length = max(len(key) for key in contents) + 1  # +1 for the space after keys
        max_value_length = max(len(format_value(value)) for value in contents.values())
    # 
    #                              CV results 
    # ---------------------------------------------------------------------------
    # Fold        Mean score        CV score      std score          Global mean 
    # ***************************************************************************       
    # Cv1          0.6789              0.6710       0.0458               0.7090
    # cv2          0.5678              0.7806       0.8907               0.9089
    # cv3          0.9807              0.6748       0.8990               0.7676
    # cv4          0.8541              0.8967       0.9087               0.6780
    # ===========================================================================
        
def summarize_inline_table(
    contents, 
    title=None, 
    header=None, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='=', 
    
    ):
    if not isinstance(contents, dict):
        raise ValueError("summarize_inline_table expects a dict of keys and values.")
    
    # Helper function to format values
    def format_value(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    
    # Initial calculations for formatting
    max_key_length = max(len(key) for key in contents) + 1  # +1 for the space after keys
    max_value_length = max(len(format_value(value)) for value in contents.values())
    
    # Adjust table width if 'max_width' is 'auto' or specified as a number
    if max_width == 'auto':
        table_width = max_key_length + max_value_length + 4  # +4 for " : " and extra space
    elif isinstance(max_width, (float, int)):
        table_width = int(max_width)
    else:
        table_width = max_key_length + max_value_length + 4  # Default behavior

    # Title and header
    title_str = f"{title.center(table_width)}" if title else ""
    top_line_str = top_line * table_width
    header_str = f"{header.center(table_width)}" if header else ""
    sub_line_str = sub_line * table_width if header else ""
    
    # Constructing content lines
    content_lines = []
    for key, value in contents.items():
        formatted_value = format_value(value)
        # Truncate long values if necessary
        space_for_value = table_width - max_key_length - 3
        if len(formatted_value) > space_for_value:
            formatted_value = formatted_value[:space_for_value-3] + "..."
    
        key_str = f"{key}"
        line = f"{key_str.ljust(max_key_length)} : {formatted_value}"
        content_lines.append(line)
    
    content_str = "\n".join(content_lines)
    bottom_line_str = bottom_line * table_width
    # Combine all parts
    table = "\n".join([title_str, top_line_str, header_str, sub_line_str,
                       content_str, bottom_line_str])
    
    return table



# def get_name_in_mixed 
# def summary2 ( model_result, **kws): 
#     if not isinstance ( model_result, dict): 
#         raise # Informative error so expect a model result to be dict 
#     # join key for easy search 
#     mixed_names = ''.join(model_results)
#     if 'estimator' not in mixed_names or 'model' not in mixed_names  or not kwargs.get('estimator_name', None): 
#         raise # Estimator name should be present 

#     if 'best_parameters' not in model_result or 'parameters'not kwargs.get('parameters', None): 
#         raise # informative error 
        
#     if 'estimator' in model_result: 
#         # now get the name who find estimator 
  
#         estimator_name = [name]
        
   


# Example usage with dummy data (This assumes you have run a GridSearchCV or RandomizedSearchCV and populated this dictionary accordingly)
# model_results = {
#     'best_parameters_': {'C': 1, 'gamma': 0.1},
#     'best_estimator_': SVC(C=1, gamma=0.1),
#     'cv_results_': {
#         'split0_test_score': [0.6789, 0.8],
#         'split1_test_score': [0.5678, 0.9],
#         'split2_test_score': [0.9807, 0.95],
#         'split3_test_score': [0.8541, 0.85],
#         'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
#         'mean_test_score': [0.770375, 0.875],
#         'std_test_score': [0.15873, 0.07],
#     },
#     'scoring': 'accuracy',
# }

# summarize_model_results(model_results)
   
def calculate_maximum_length( report_data, max_value_length = 50 ): 
    # Calculate the maximum key length for alignment
    max_key_length = max(len(key) for key in report_data.keys())
    # calculate the maximum values length 
    max_val_length = 0 
    for value in report_data.values (): 
        if isinstance ( value, (int, float, np.integer, np.floating)): 
            value = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 
        if isinstance ( value, pd.Series): 
            value = format_series ( value)
        
        if max_val_length < len(value): 
            max_val_length = len(value) 
            
    return max_key_length, max_val_length
       
def format_value( value ): 
    value_str =''
    if isinstance(value, (int, float, np.integer, np.floating)): 
        value_str = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 
        
    return str(value_str)  

def format_report(report_data, report_title=None):
    # Calculate the maximum key length for alignment
    max_key_length, max_val_length  = calculate_maximum_length( report_data)
    
    # Prepare the report title and frame lines
    line_length = max_key_length + max_val_length + 4 # Adjust for key-value spacing and aesthetics
    top_line = "=" * line_length
    subsection_line = '-'* line_length
    bottom_line = "=" * line_length
    
    # Construct the report string starting with the top frame line
    report_str = f"{top_line}\n"
    
    # Add the report title if provided, centered within the frame
    if report_title:
        report_str += f"{report_title.center(line_length)}\n{subsection_line}\n"
    
    # Add each key-value pair to the report
    for key, value in report_data.items():
        # Format numeric values with four decimal places
        if isinstance ( value, (int, float, np.integer, np.floating)): 
            formatted_value = format_value ( value )
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        elif isinstance ( value, pd.Series): 
            formatted_value = format_series(value)
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        elif isinstance(value, pd.DataFrame): 
            formatted_value = dataframe_key_format(
                key, value, max_key_length=max_key_length)
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
            
        elif isinstance ( value, str) and len(value) > line_length: 
            # consider as long text 
            formatted_value = format_text(
                value, key_length=max_key_length, max_char_text= max_val_length)
            
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        else :    
        # formatted_value = _format_values ( value ) if isinstance(value, (
        #     int, float, np.integer, np.floating)) else ( 
        #     _format_series(value) if isinstance (value, pd.Series ) else value 
        #     ) 
        # Construct the line with key and value, aligning based on the longest key
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
    
    # Add the bottom frame line
    report_str += bottom_line
    
    return report_str

# Refactor summarize_model_results and rather than use long string display with
#  table contenst which is not professionnaly , I 
# write helper function named "format_dataframe' below ( use this function , dont rewrite).
#  contruct a  dataframe with columns ["Fold",  "Mean score" ,  "CV score" ,  "std score ",  "Global mean" ]
# from the above code  and use this dataframe to format the result via format_dataframe to 
# display the result as: 

#                              Model Results 
# ---------------------------------------------------------------------------
# Fold        Mean score        CV score      std score          Global mean 
# ***************************************************************************       
# Cv1          0.6789              0.6710       0.0458               0.7090
# cv2          0.5678              0.7806       0.8907               0.9089
# cv3          0.9807              0.6748       0.8990               0.7676
# cv4          0.8541              0.8967       0.9087               0.6780
# ===========================================================================

def summarize_model_results(
        model_results, title=None, 
        top_line='=', sub_line='-', bottom_line='=', 
        ):
    title = title or "Model Results"
    keys = ["Fold", "Mean score", "CV score", "std score", "Global mean"]
    
    # Extracting required information from model_results
    cv_results = model_results['cv_results_']
    nCV = len({k for k in cv_results.keys() if k.startswith('split')}) // len(cv_results['params'])
    
    # Prepare data for DataFrame
    data = []
    for i in range(nCV):
        mean_score = np.mean([cv_results[f'split{j}_test_score'][i] for j in range(nCV)])
        cv_scores = [cv_results[f'split{j}_test_score'][i] for j in range(nCV)]
        std_score = np.std(cv_scores)
        global_mean = np.nanmean(cv_scores)
        
        fold_data = [f"Cv{i+1}", mean_score, cv_scores[i], std_score, global_mean]
        data.append(fold_data)
    
    # Creating DataFrame
    df = pd.DataFrame(data, columns=keys)
    
    # Formatting DataFrame using the provided 'format_dataframe' function
    formatted_table = format_dataframe(
        df, top_line=top_line, 
        sub_line=sub_line, 
        bottom_line=bottom_line)
    
    return  len(formatted_table.split('\n')[0]) # 

    # # Print title and formatted table
    # print(title.center(len(formatted_table.split('\n')[0])))
    # print(formatted_table)
    
    

# Assuming 'model_results' is a dictionary obtained from GridSearchCV or RandomizedSearchCV
# Example usage
model_results = {
    'best_parameters_': {'C': 1, 'gamma': 0.1},
    'best_estimator_': "SVC",
    'cv_results_': {
        'split0_test_score': [0.6789, 0.8],
        'split1_test_score': [0.5678, 0.9],
        'split2_test_score': [0.9807, 0.95],
        'split3_test_score': [0.8541, 0.85],
        'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
    },
    'scoring': 'accuracy',
}

# Note: The actual 'model_results' would be the output from a GridSearchCV or RandomizedSearchCV operation.
# The 'format_dataframe' function needs to be defined as given, and pandas must be installed.

# def summarize_model_results(model_results, max_width ='auto', title=None):
    
#     title = title or "Model Results"
    
    
#     keys = ["Fold",  "Mean score" ,  "CV score" ,  "std score ",  "Global mean" ]
    
#     estimator_name = type(model_results['best_estimator_']).__name__
#     best_params = model_results['best_parameters_']
#     cv_results = model_results['cv_results_']
#     nCV = len({k for k in cv_results.keys() if k.startswith('split')}) // len(cv_results['params'])
#     scoring = model_results.get('scoring', 'Unknown scoring')

#     header = " " * 29 + "Results\n" + "=" * 74 + "\n" + " " * 30 + "Main\n" + "-" * 74
#     model_summary = f"\nEstimator         : {estimator_name}\nBest parameters   : {best_params}\nnCV               : {nCV}\nscoring           : {scoring}\n" + "-" * 74 + "\n" + " " * 29 + "CV results\n" + "-" * 74

#     cv_results_str = "Fold        Mean score        CV score      std score          Global mean\n" + "*" * 75
#     global_means = []

#     for i in range(nCV):
#         mean_score = np.mean([cv_results[f'split{j}_test_score'][i] for j in range(nCV)])
#         cv_scores = [cv_results[f'split{j}_test_score'][i] for j in range(nCV)]
#         std_score = np.std(cv_scores)
#         global_mean = np.nanmean(cv_scores)
#         global_means.append(global_mean)
        
#         cv_results_str += f"\nCv{i+1:<10}{mean_score:<18.4f}{cv_scores[i]:<15.4f}{std_score:<18.4f}{global_mean:<15.4f}"

#     overall_global_mean = np.nanmean(global_means)
#     cv_results_str += f"\n{' ' * 59}{'=' * 15}\n{'Global Mean:':<59}{overall_global_mean:.4f}"

#     summary = "\n".join([header, model_summary, cv_results_str, "=" * 75])
    
#     print(summary)

def format_dataframe(
    df, 
    max_long_text_char=50,  
    top_line='=', 
    sub_line='-', 
    bottom_line='=', 
     ):
    # Calculate the max length for the index
    max_index_length = max([len(str(index)) for index in df.index])
    
    # Calculate max length for each column including the column name, 
    # and account for truncation
    max_col_lengths = {
        col: max(len(col), max(df[col].astype(str).apply(
            lambda x: len(x) if len(x) <= max_long_text_char else max_long_text_char + 3)))
        for col in df.columns
    }
    
    # Adjust column values for truncation and formatting
    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x if len(x) <= max_long_text_char else x[:max_long_text_char] + "...")
        max_col_lengths[col] = max(max_col_lengths[col], df[col].str.len().max())
    
    # Construct the header with padding for alignment
    header = " ".join([f"{col:>{max_col_lengths[col]}}" for col in df.columns])
    # Calculate total table width
    table_width = sum(max_col_lengths.values()) + len(max_col_lengths) - 1 + max_index_length + 1
    # Construct the separators
    top_header_border = top_line * table_width
    separator = sub_line * table_width
    top_bottom_border = top_line * table_width
    
    # Construct each row
    rows = []
    for index, row in df.iterrows():
        row_str = f"{str(index).ljust(max_index_length)} " + " ".join(
            [f"{value:>{max_col_lengths[col]}}" for col, value in row.iteritems()])
        rows.append(row_str)
    
    # Combine all parts
    table = f"{top_header_border}\n{' ' * (max_index_length + 1)}{header}\n{separator}\n" + "\n".join(
        rows) + f"\n{top_bottom_border}"
    
    return table



def format_key(key, max_length=None, include_colon=False, alignment='left'):
    """
    Formats a key string according to the specified parameters.

    Parameters:
    - key (str): The key to format.
    - max_length (int, optional): The maximum length for the formatted key. 
      If None, it's calculated based on the key length.
    - include_colon (bool): Determines whether a colon and space should be 
      added after the key.
    - alignment (str): Specifies the alignment of the key. Expected values are 
     'left' or 'right'.

    Returns:
    - str: The formatted key string.
    # Example usage:
    print(format_key("ExampleKey", 20, include_colon=True, alignment='left'))
    print(format_key("Short", max_length=10, include_colon=False, alignment='right'))
    
    """
    # Calculate the base length of the key, optionally including the colon and space
    base_length = len(key) + (2 if include_colon else 0)
    
    # Determine the final max length, using the base length if no max_length is provided
    final_length = max_length if max_length is not None else base_length
    
    # Construct the key format string
    key_format = f"{key}:" if include_colon else key
    
    # Apply the specified alignment and padding
    formatted_key = ( f"{key_format: <{final_length}}" if alignment == 'left' 
                     else f"{key_format: >{final_length}}"
                     )
    
    return formatted_key

def dataframe_key_format(key, df, max_key_length=None, max_text_char=50):
    # Format the key with a colon, using the provided or calculated max key length
    formatted_key = format_key(key, max_length=max_key_length,
                               include_colon=True, alignment='left')
    
    # Format the DataFrame according to specifications
    formatted_df = format_dataframe(df, max_long_text_char=max_text_char)
    
    # Split the formatted DataFrame into lines for alignment adjustments
    df_lines = formatted_df.split('\n')
    
    # Determine the number of spaces to prepend to each line based on the 
    # length of the formatted key
    # Adding 1 additional space for alignment under the formatted key
    space_prefix = ' ' * (len(formatted_key) + 1)
    
    # Prepend spaces to each line of the formatted DataFrame except for the 
    # first line (it's already aligned)
    aligned_df = space_prefix + df_lines[0] + '\n' + '\n'.join(
        [space_prefix + line for line in df_lines[1:]])
    
    # Combine the formatted key with the aligned DataFrame
    result = f"{formatted_key}\n{aligned_df}"
    
    return result

# Assuming format_key and format_dataframe are correctly implemented as discussed in previous steps
# Example usage would be as follows (after defining df):
# print(dataframe_key_format("Your Key Here", df))
       
# def dataframe_key_format( key, df,  max_key_length = None, max_text_char=50  ):
    
#     formatted_key = format_key ( key, max_key_length, include_colon= True ) 
    
#     formatted_df = format_dataframe(df, max_long_text_char= max_text_char)
    
    # once the key is formatted. Note the format_dataframe construct 
    # the formatted_df like below : 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             col1     col2     col3 
    # ----------------------------------
    # index1   value11  value12  value13
    # index2   value21  value12  value13
    # index3   value31  value13  value13
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Then now move the formatted_df based on the formatted key length by 
    # mentionned the key like below:
     
    # [formatted_key] 
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                         col1     col2     col3 
    #                ----------------------------------
    #                index1   value11  value12  value13
    #                index2   value21  value12  value13
    #                index3   value31  value13  value13
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # for instance if the formatted key = key            : 
    # formatted key with df should be : 
        
    # key            :
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                         col1     col2     col3 
    #                ----------------------------------
    #                index1   value11  value12  value13
    #                index2   value21  value12  value13
    #                index3   value31  value13  value13

def format_dict(dct, ):
    # # Example usage with a mixed dictionary
    # mixed_dict = {
    #     "a": "apple",
    #     "b": 2,
    #     "c": 3.5,
    #     "d": float('nan'),
    #     "e": "banana"
    # }
    # print(format_dict(mixed_dict))

    # # Example usage with a numeric dictionary
    # numeric_dict = {
    #     "one": 1,
    #     "two": 2,
    #     "three": 3,
    #     "four": float('nan'),
    #     "five": 5
    # }
    # print(format_dict(numeric_dict))
    
    # Initialize counts
    num_values_count = 0
    non_num_values_count = 0
    exist_nan = False
    
    # Iterate through the dictionary values to categorize them
    for value in dct.values():
        if isinstance(value, (int, float)):
            num_values_count += 1
            if isinstance(value, float) and np.isnan(value):
                exist_nan = True
        else:
            non_num_values_count += 1
            
    # Determine dtype description
    dtype_description = ( 
        "numeric" if non_num_values_count == 0 else "mixed" 
        if num_values_count > 0 else "non-numeric"
        )
    
    # Calculate mean for numeric values if any, ignoring NaNs
    if num_values_count > 0:
        numeric_values = np.array([value for value in dct.values() if isinstance(
            value, (int, float))], dtype=float)
        mean_value = np.nanmean(numeric_values)
        summary_str = (
            "Dict ~ len:{} - values: <mean: {:.4f} - numval: {}"
            " - nonnumval: {} - dtype: {} - exist_nan:"
            " {}>").format(
            len(dct), mean_value, num_values_count, non_num_values_count,
            dtype_description, exist_nan
        )
    else:
        summary_str = ( 
            "Dict ~ len:{} - values: <numval: {} - nonnumval: {}"
            " - dtype: {} - exist_nan: {}>").format(
            len(dct), num_values_count, non_num_values_count, dtype_description,
            exist_nan
        )

    return summary_str


def format_list(lst, ):
    # Check if all elements in the list are numeric (int or float)
    all_numeric = all(isinstance(x, (int, float)) for x in lst)
    exist_nan = any(np.isnan(x) for x in lst if isinstance(x, float))

    if all_numeric:
        # Calculate mean for numeric list, ignoring NaNs
        numeric_values = np.array(lst, dtype=float)  # Convert list to NumPy array for nanmean calculation
        mean_value = np.nanmean(numeric_values)
        arr_str = "List ~ len:{} - values: < mean: {:.4f} - dtype: numeric - exist_nan: {}>".format(
            len( lst), mean_value, exist_nan
        )
    else:
        # For mixed or non-numeric lists, calculate the count of numeric and non-numeric values
        num_values_count = sum(isinstance(x, (int, float)) for x in lst)
        non_num_values_count = len(lst) - num_values_count
        dtype_description = "mixed" if not all_numeric else "non-numeric"
        arr_str = "List ~ len:{} - values: <numval: {} - nonnumval: {} - dtype: {} - exist_nan: {}>".format(
            len( lst), num_values_count, non_num_values_count, dtype_description, exist_nan
        )

    return arr_str


def format_array(arr, ):
    # # Example usage with a numeric array
    # numeric_arr = np.array([1, 2, 3, np.nan, 5])
    # print(format_array(numeric_arr))

    # # Example usage with a non-numeric (mixed) array
    # # This will not provide a meaningful summary for non-numeric data types,
    # # as the logic for non-numeric arrays would need to be more complex and might require pandas
    # mixed_arr = np.array(["apple", 2, 3.5, np.nan, "banana"], dtype=object)
    # print(format_array(mixed_arr))


    arr_str = ""
    # Check if the array contains NaN values; works only for numeric arrays
    exist_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.number) else False

    if np.issubdtype(arr.dtype, np.number):
        # Formatting string for numeric arrays
        arr_str = "Array ~ {}values: <mean: {:.4f} - dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            np.nanmean(arr),  # Use nanmean to calculate mean while ignoring NaNs
            arr.dtype.name,
            ' - exist_nan:True' if exist_nan else ''
        )
    else:
        # Attempt to handle mixed or non-numeric data without pandas,
        # acknowledging that numpy arrays are typically of a single data type
        # Here we consider the array non-numeric and don't attempt to summarize numerically
        dtype_description = "mixed" if arr.dtype == object else arr.dtype.name
        arr_str = "Array ~ {}values: <dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            dtype_description,
            ' - exist_nan:True' if exist_nan else ''
        )

    return arr_str




def format_text(text, key=None, key_length=15, max_char_text=50):
    # Example usage:
    # text_example = "This is an example text that is supposed to wrap around after a 
    # certain number of characters, demonstrating the function."

    # # # Test with key and default key_length
    # print(format_text(text_example, key="ExampleKey"))

    # # Test with key and custom key_length
    # print(format_text(text_example, key="Key", key_length=10))

    # # Test without key but with key_length
    # print(format_text(text_example, key_length=5))

    # # Test without key and key_length
    # print(format_text(text_example))
    
    if key is not None:
        # If key_length is None, use the length of the key + 1 for the space after the key
        if key_length is None:
            key_length = len(key) + 1
        key_str = f"{key.ljust(key_length)}: "
    elif key_length is not None:
        # If key is None but key_length is specified, use spaces
        key_str = " " * key_length + ": "
    else:
        # If both key and key_length are None, there's no key part
        key_str = ""
    
    # Adjust max_char_text based on the length of the key part
    effective_max_char_text = max_char_text - len(key_str) + 2 if key_str else max_char_text

    formatted_text = ""
    while text:
        # If the remaining text is shorter than the effective max length, or if there's no key part, add it as is
        if len(text) <= effective_max_char_text or not key_str:
            formatted_text += key_str + text
            break
        else:
            # Find the space to break the line, ensuring it doesn't exceed effective_max_char_text
            break_point = text.rfind(' ', 0, effective_max_char_text)
            if break_point == -1:  # No spaces found, force break
                break_point = effective_max_char_text
            # Add the line to formatted_text
            formatted_text += key_str + text[:break_point].rstrip() + "\n"
            # Remove the added part from text
            text = text[break_point:].lstrip()
            # After the first line, the key part is just spaces
            key_str = " " * len(key_str)
    
    return formatted_text


def format_series(series):
    # Example usage:
    # For a numeric series
    # numeric_series = pd.Series([1, 2, 3, np.nan, 5, 6], name='NumericSeries')
    # print(_format_series(numeric_series))

    # # For a non-numeric series
    # non_numeric_series = pd.Series(['apple', 'banana', np.nan, 'cherry', 'date', 'eggfruit', 'fig'], name='FruitSeries')
    # print(_format_series(non_numeric_series))

    series_str = ''
    if not isinstance ( series, pd.Series):
        return series 
    
    if series.dtype.kind in 'biufc' and len(series) < 7:  # Check if series is numeric and length < 7
        series_str = "Series ~ {}values: <mean: {:.4f} - length: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            np.mean(series.values), 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    else:  # Handle non-numeric series or series with length >= 7
        num_values = series.apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x)).sum()
        non_num_values = len(series) - num_values - series.isnull().sum()  # Exclude NaNs from non-numeric count
        series_str = "Series ~ {}values: <numval: {} - nonnumval: {} - length: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            num_values, 
            non_num_values, 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    return series_str


# Example usage
if __name__ == "__main__":
    # Example usage:
    report_data = {
        'Total Sales': 123456.789,
        'Average Rating': 4.321,
        'Number of Reviews': 987,
        'Key with long name': 'Example text'
    }

    report_title = 'Monthly Sales Report'
    formatted_report = format_report(report_data, report_title)
    print(formatted_report)
    
    # report = Report(title="Example Report")
    # report.add_data_ops_report({
    #     "key1": np.random.random(),
    #     "key2": "value2",
    #     "key3": pd.Series([1, 2, 3], name="series_name"),
    #     "key4": pd.DataFrame(np.random.rand(4, 3), columns=["col1", "col2", "col3"])
    # })
    # print(report)
    
    # # Creating an example DataFrame
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