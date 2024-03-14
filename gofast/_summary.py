# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:29:22 2024

@author: Daniel
"""
import numpy as np
import pandas as pd

class Summary:
    def __init__(self, data):
        """
        Initialize with either a pandas DataFrame or a 'bunch' object (similar to sklearn's Bunch).
        The object should contain either the data for summary1 or model results for summary2.
        """
        self.data = data

    def summary1(self):
        """
        Display basic statistics of a DataFrame, with numerical values rounded to 4 decimal places.
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
            desc = self.data.describe(include='all').applymap(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
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
    