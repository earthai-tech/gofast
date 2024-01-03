# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Learning utilities for data transformation, 
model learning and inspections. 
"""
from __future__ import annotations 
import os 
import copy 
import inspect 
import hashlib 
import tarfile 
import warnings 
import pickle 
import joblib
import datetime 
import shutil
from pprint import pprint  
from six.moves import urllib 
from collections import Counter 
import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder,RobustScaler ,OrdinalEncoder 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,  LabelBinarizer
from sklearn.preprocessing import LabelEncoder,Normalizer, PolynomialFeatures 

from .._gofastlog import gofastlog
from .._typing import List, Tuple, Any, Dict,  Optional,Union, Iterable,Series 
from .._typing import _T, _F, _Sub,  ArrayLike, NDArray,DType, DataFrame, Set  
from ._dependency import import_optional_dependency
from ..exceptions import ParameterNumberError, EstimatorError, DatasetError
from ..decorators import deprecated              
from .funcutils import _assert_all_types, _isin,  is_in_if,  ellipsis2false
from .funcutils import savepath_, smart_format, str2columns, is_iterable
from .funcutils import  is_classification_task, to_numeric_dtypes, fancy_printer
from .validator import get_estimator_name, check_array, check_consistent_length
from .validator import  _is_numeric_dtype,  _is_arraylike_1d 
from .validator import  is_frame, build_data_if

_logger = gofastlog().get_gofast_logger(__name__)


__all__=[ 
    "evaluate_model",
    "select_features", 
    "get_global_score", 
    "get_correlated_features", 
    "find_features_in", 
    "categorize_target", 
    "resampling", 
    "bin_counting", 
    "labels_validator", 
    "projection_validator", 
    "rename_labels_in" , 
    "soft_imputer", 
    "soft_scaler", 
    "select_feature_importances", 
    "load_saved_model", 
    "make_pipe",
    "build_data_preprocessor", 
    "bi_selector", 
    "get_target", 
    "export_target",  
    "stats_from_prediction", 
    "fetch_tgz", 
    "fetch_model", 
    "load_csv", 
    "features_in", 
    "split_train_test_by_id", 
    "split_train_test", 
    "discretize_categories", 
    "stratify_categories", 
    "serialize_data", 
    "load_dumped_data", 
    "soft_data_split",
    "laplace_smoothing"
    ]


_scorers = { 
    "classification_report":classification_report,
    'precision_recall': precision_recall_curve,
    "confusion_matrix":confusion_matrix,
    'precision': precision_score,
    "accuracy": accuracy_score,
    "mse":mean_squared_error, 
    "recall": recall_score, 
    'auc': roc_auc_score, 
    'roc': roc_curve, 
    'f1':f1_score,
    }

_estimators ={
        'dtc': ['DecisionTreeClassifier', 'dtc', 'dec', 'dt'],
        'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
        'sdg': ['SGDClassifier','sdg', 'sd', 'sdg'],
        'knn': ['KNeighborsClassifier','knn', 'kne', 'knr'],
        'rdf': ['RandomForestClassifier', 'rdf', 'rf', 'rfc',],
        'ada': ['AdaBoostClassifier','ada', 'adc', 'adboost'],
        'vtc': ['VotingClassifier','vtc', 'vot', 'voting'],
        'bag': ['BaggingClassifier', 'bag', 'bag', 'bagg'],
        'stc': ['StackingClassifier','stc', 'sta', 'stack'],
    'xgboost': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm', 'xgb'], 
     'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
     'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
        }  
#------

def codify_variables (
    arr, /, 
    columns: list =None, 
    func: _F=None, 
    categories: dict=None, 
    get_dummies:bool=..., 
    parse_cols:bool =..., 
    return_cat_codes:bool=... 
    ) -> DataFrame: 
    """ Encode multiple categorical variables in a dataset. 

    Parameters 
    -----------
    arr: pd.DataFrame, ArrayLike, dict 
       DataFrame or Arraylike. If simple array is passed, specify the 
       columns argumment to create a dataframe. If a dictionnary 
       is passed, it should be convert to a dataframe. 
       
    columns: list,
       List of the columns to encode the labels 
       
    func: callable, 
       Function to apply the label accordingly. Label must be included in 
       the columns values.
       
    categories: dict, Optional 
       Dictionnary of column names(`key`) and labels (`values`) to 
       map the labels.  
       
    get_dummies: bool, default=False 
      returns a new encoded DataFrame  with binary columns 
      for each category within the specified categorical columns.

    parse_cols: bool, default=False
      If `columns` parameter is listed as string, `parse_cols` can defaultly 
      constructs an iterable objects. 
    
    return_cat_codes: bool, default=False 
       return the categorical codes that used for mapping variables. 
       if `func` is applied, mapper returns an empty dict. 
       
    Return
    -------
    df: New encoded Dataframe 
    
    Examples
    ----------
    >>> from gofast.tools.mlutils import codify_variables 
    >>> # Sample dataset with categorical variables
    >>> data = {'Height': [152, 175, 162, 140, 170], 
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
        'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small'],
        'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Triangle'], 
        'Weight': [80, 75, 55, 61, 70]
    }
    # List of categorical columns to one-hot encode
    categorical_columns = ['Color', 'Size', 'Shape']
    >>> df_encoded = codify_variables (data)
    >>> df_encoded.head(2) 
    Out[1]: 
       Height  Weight  Color  Size  Shape
    0     152      80      2     2      0
    1     175      75      0     0      1
    >>> # nw return_map codes 
    >>> df_encoded , map_codes =codify_variables (
        data, return_cat_codes =True )
    >>> map_codes 
    Out[2]: 
    {'Color': {2: 'Red', 0: 'Blue', 1: 'Green'},
     'Size': {2: 'Small', 0: 'Large', 1: 'Medium'},
     'Shape': {0: 'Circle', 1: 'Square', 2: 'Triangle'}}
    >>> def cat_func (x ): 
        # 2: 'Red', 0: 'Blue', 1: 'Green'
        if x=='Red': 
            return 2 
        elif x=='Blue': 
            return 0
        elif x=='Green': 
            return 1 
        else: return x 
    >>> df_encoded =codify_variables (data, func= cat_func)
    >>> df_encoded.head(3) 
    Out[3]: 
       Height  Color    Size     Shape  Weight
    0     152      2   Small    Circle      80
    1     175      0   Large    Square      75
    2     162      1  Medium  Triangle      55
    >>> 
    >>> # Perform one-hot encoding
    >>> df_encoded = codify_variables (data, get_dummies=True )
    >>> df_encoded.head(3)
    Out[4]: 
       Height  Weight  Color_Blue  ...  Shape_Circle  Shape_Square  Shape_Triangle
    0     152      80           0  ...             1             0               0
    1     175      75           1  ...             0             1               0
    2     162      55           0  ...             0             0               1
    [3 rows x 11 columns]
    >>> codify_variables (data, categories ={'Size': ['Small', 'Large',  'Medium']})
    Out[5]: 
       Height  Color     Shape  Weight  Size
    0     152    Red    Circle      80     0
    1     175   Blue    Square      75     1
    2     162  Green  Triangle      55     2
    3     140    Red    Circle      61     2
    4     170   Blue  Triangle      70     0
    """
    get_dummies, parse_cols, return_cat_codes = ellipsis2false(
        get_dummies, parse_cols, return_cat_codes )
    # build dataframe if arr is passed rather 
    # than a dataframe 
    df = build_data_if( arr, to_frame =True, force=True, input_name ='col',
                        raise_warning='silence'  )
    # now check integrity 
    df = to_numeric_dtypes( df )
    if columns is not None: 
        columns = list( 
            is_iterable(columns, exclude_string =True, transform =True, 
                              parse_string= parse_cols 
                              )
                       )
        df = select_features(df, features = columns )
        
    map_codes ={}     
    if get_dummies :
        # Perform one-hot encoding
        # We use the pd.get_dummies() function from the pandas library 
        # to perform one-hot encoding on the specified columns
        return ( ( pd.get_dummies(df, columns=columns) , map_codes )
                  if return_cat_codes else ( 
                          pd.get_dummies(df, columns=columns) ) 
                )
    # ---work with category -------- 
    # if categories is Note , get auto numeric and 
    # categoric variablees 
    num_columns, cat_columns = bi_selector (df ) 
    #apply function if 
    # function is given 
    
    if func is not None: 
        # just get only the columns 
        if not callable (func): 
            raise TypeError("Expect an universal function."
                            f" Got {type(func).__name__!r}")
        if len(cat_columns)==0: 
            # no categorical data func. 
            msg =("No categorical data detected. To transform numeric"
                " values to labels, use `gofast.tools.smart_label_classifier`"
                " or `gofast.tools.categorize_target` instead.")
            warnings.warn (msg) 
            return df 
        
        for col in  cat_columns: 
            df[col]= df[col].apply (func ) 

        return (df, map_codes) if return_cat_codes else df 
 
    if categories is None: 
        categories ={}
        for col in cat_columns: 
            categories[col] = list(np.unique (df[col] ))
            
    # categories should be a mapping data 
    if not isinstance ( categories, dict ): 
        raise TypeError("Expect a dictionnary {`column name`:`labels`}"
                        "to categorize data.")
        
    for col, values  in  categories.items():
        if col not in df.columns:
            print(col)
            continue  
        values = is_iterable(
            values, exclude_string=True, transform =True )
        df[col] = pd.Categorical (df[col], categories = values, ordered=True )
        # df[col] = df[col].astype ('category')
        val=df[col].cat.codes
        temp_col = col + '_col'
        df[temp_col] = val 
        map_codes[col] =  dict(zip(df[col].cat.codes, df[col]))
        # drop prevous col in the data frame 
        df.drop ( columns =[col], inplace =True ) 
        # rename the tem colum 
        # to take back to pandas 
        df.rename ( columns ={temp_col: col }, inplace =True ) 
        
    return (df, map_codes) if return_cat_codes else df 

def resampling( 
    X, 
    y, 
    kind ='over', 
    strategy ='auto', 
    random_state =None, 
    verbose: bool=..., 
    **kws
    ): 
    """ Combining Random Oversampling and Undersampling 
    
    Resampling involves creating a new transformed version of the training 
    dataset in which the selected examples have a different class distribution.
    This is a simple and effective strategy for imbalanced classification 
    problems.

    Applying re-sampling strategies to obtain a more balanced data 
    distribution is an effective solution to the imbalance problem. There are 
    two main approaches to random resampling for imbalanced classification; 
    they are oversampling and undersampling.

    - Random Oversampling: Randomly duplicate examples in the minority class.
    - Random Undersampling: Randomly delete examples in the majority class.
        
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y: array-like of shape (n_samples, ) 
        Target vector where `n_samples` is the number of samples.
    kind: str, {"over", "under"} , default="over"
      kind of sampling to perform. ``"over"`` and ``"under"`` stand for 
      `oversampling` and `undersampling` respectively. 
      
    strategy : float, str, dict, callable, default='auto'
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
          
    random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.

            - If int, ``random_state`` is the seed used by the random number
              generator;
            - If ``RandomState`` instance, random_state is the random number
              generator;
            - If ``None``, the random number generator is the ``RandomState``
              instance used by ``np.random``.
              
    verbose: bool, default=False 
      Display the counting samples 
      
    Returns 
    ---------
    X, y : NDarray, Arraylike 
        Arraylike sampled 
    
    Examples 
    --------- 
    >>> import gofast as gf 
    >>> from gofast.tools.mlutils import resampling 
    >>> data, target = gf.fetch_data ('bagoue analysed', as_frame =True) 
    >>> data.shape, target.shape 
    >>> data_us, target_us = resampling (data, target, kind ='under',
                                         verbose=True)
    >>> data_us.shape, target_us.shape 
    Counters: Auto      
                         Raw counter y: Counter({0: 232, 1: 112})
               UnderSampling counter y: Counter({0: 112, 1: 112})
    Out[43]: ((224, 8), (224,))
    
    """
    msg =(" `imblearn` is the shorthand of the package 'imbalanced-learn'."
          " Use `pip install imbalanced-learn` instead.")
    import_optional_dependency ("imblearn", extra = msg )
    kind = str(kind).lower() 
    if kind =='under': 
        from imblearn.under_sampling import RandomUnderSampler
        rsampler = RandomUnderSampler(sampling_strategy=strategy, 
                                      random_state = random_state ,
                                      **kws)
    else:  
        from imblearn.over_sampling import RandomOverSampler 
        rsampler = RandomOverSampler(sampling_strategy=strategy, 
                                     random_state = random_state ,
                                     **kws
                                     )
    Xs, ys = rsampler.fit_resample(X, y)
    
    if ellipsis2false(verbose)[0]: 
        print("{:<20}".format(f"Counters: {strategy.title()}"))
        print( "{:>35}".format( "Raw counter y:") , Counter (y))
        print( "{:>35}".format(f"{kind.title()}Sampling counter y:"), Counter (ys))
        
    return Xs, ys 

def bin_counting(
    data: DataFrame, 
    bin_columns: str|List[str, ...], 
    tname:str|Series[int], 
    odds="N+", 
    return_counts: bool=...,
    tolog: bool=..., 
    ): 
    """ Bin counting categorical variable and turn it into probabilistic 
    ratio.
    
    Bin counting is one of the perennial rediscoveries in machine learning. 
    It has been reinvented and used in a variety of applications, from ad 
    click-through rate prediction to hardware branch prediction [1]_, [2]_ 
    and [3]_.
    
    Given an input variable X and a target variable Y, the odds ratio is 
    defined as:
        
    .. math:: 
        
        odds ratio = \frac{ P(Y = 1 | X = 1)/ P(Y = 0 | X = 1)}{
            P(Y = 1 | X = 0)/ P(Y = 0 | X = 0)}
          
    Probability ratios can easily become very small or very large. The log 
    transform again comes to our rescue. Anotheruseful property of the 
    logarithm is that it turns a division into a subtraction. To turn 
    bin statistic probability value to log, set ``uselog=True``.
    
    Parameters 
    -----------
    data: dataframe 
       Data containing the categorical values. 
       
    bin_columns: str or list 
       The columns to applied the bin_countings 
       
    tname: str, pd.Series
      The target name for which the counting is operated. If series, it 
      must have the same length as the data. 
      
    odds: str , {"N+", "N-", "log_N+"}: 
        The odds ratio of bin counting to fill the categorical. ``N+`` and  
        ``N-`` are positive and negative probabilistic computing. Whereas the
        ``log_N+`` is the logarithm odds ratio useful when value are smaller 
        or larger. 
        
    return_counts: bool, default=True 
      return the bin counting dataframes. 
  
    tolog: bool, default=False, 
      Apply the logarithm to the output data ratio. Indeed, Probability ratios 
      can easily  become very small or very large. For instance, there will be 
      users who almost never click on ads, and perhaps users who click on ads 
      much more frequently than not.) The log transform again comes to our  
      rescue. Another useful property of the logarithm is that it turns a 
      division 

    Returns 
    --------
    d: dataframe 
       Dataframe transformed or bin-counting data
       
    Examples 
    ---------
    >>> import gofast as gf 
    >>> from gofast.tools.mlutils import bin_counting 
    >>> X, y = gf.fetch_data ('bagoue analysed', as_frame =True) 
    >>> # target binarize 
    >>> y [y <=1] = 0;  y [y > 0]=1 
    >>> X.head(2) 
    Out[7]: 
          power  magnitude       sfi      ohmS       lwi  shape  type  geol
    0  0.191800  -0.140799 -0.426916  0.386121  0.638622    4.0   1.0   3.0
    1 -0.430644  -0.114022  1.678541 -0.185662 -0.063900    3.0   2.0   2.0
    >>>  bin_counting (X , bin_columns= 'geol', tname =y).head(2)
    Out[8]: 
          power  magnitude       sfi      ohmS  ...  shape  type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  0.386121  ...    4.0   1.0  0.656716           1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...    3.0   2.0  0.219251           0
    [2 rows x 9 columns]
    >>>  bin_counting (X , bin_columns= ['geol', 'shape', 'type'], tname =y).head(2)
    Out[10]: 
          power  magnitude       sfi  ...      type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.267241  0.656716           1
    1 -0.430644  -0.114022  1.678541  ...  0.385965  0.219251           0
    [2 rows x 9 columns]
    >>> df = pd.DataFrame ( pd.concat ( [X, pd.Series ( y, name ='flow')],
                                       axis =1))
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'], 
                      tname ="flow", tolog=True).head(2)
    Out[12]: 
          power  magnitude       sfi      ohmS  ...     shape      type      geol  flow
    0  0.191800  -0.140799 -0.426916  0.386121  ...  0.828571  0.364706  1.913043     1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...  0.364865  0.628571  0.280822     0
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'],odds ="N-", 
                      tname =y, tolog=True).head(2)
    Out[13]: 
          power  magnitude       sfi  ...      geol  flow  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.522727     1           1
    1 -0.430644  -0.114022  1.678541  ...  3.560976     0           0
    [2 rows x 10 columns]
    >>> bin_counting (df , bin_columns= "geol",tname ="flow", tolog=True,
                      return_counts= True )
    Out[14]: 
         flow  no_flow  total_flow        N+        N-     logN+     logN-
    3.0    44       23          67  0.656716  0.343284  1.913043  0.522727
    2.0    41      146         187  0.219251  0.780749  0.280822  3.560976
    0.0    18       43          61  0.295082  0.704918  0.418605  2.388889
    1.0     9       20          29  0.310345  0.689655  0.450000  2.222222

    References 
    -----------
    .. [1] Yeh, Tse-Yu, and Yale N. Patt. Two-Level Adaptive Training Branch 
           Prediction. Proceedings of the 24th Annual International 
           Symposium on Microarchitecture (1991):51–61
           
    .. [2] Li, Wei, Xuerui Wang, Ruofei Zhang, Ying Cui, Jianchang Mao, and 
           Rong Jin.Exploitation and Exploration in a Performance Based Contextual 
           Advertising System. Proceedings of the 16th ACM SIGKDD International
           Conference on Knowledge Discovery and Data Mining (2010): 27–36
           
    .. [3] Chen, Ye, Dmitry Pavlov, and John _F. Canny. “Large-Scale Behavioral 
           Targeting. Proceedings of the 15th ACM SIGKDD International 
           Conference on Knowledge Discovery and Data Mining (2009): 209–218     
    """
    # assert everything
    if not is_frame (data, df_only =True ):
        raise TypeError(f"Expect dataframe. Got {type(data).__name__!r}")
    
    if not _is_numeric_dtype(data, to_array= True): 
        raise TypeError ("Expect data with encoded categorical variables."
                         " Please check your data.")
    if hasattr ( tname, '__array__'): 
        check_consistent_length( data, tname )
        if not _is_arraylike_1d(tname): 
            raise TypeError (
                 "Only one dimensional array is allowed for the target.")
        # create fake bin target 
        if not hasattr ( tname, 'name'): 
            tname = pd.Series (tname, name ='bin_target')
        # concatenate target 
        data= pd.concat ( [ data, tname], axis = 1 )
        tname = tname.name  # take the name 
        
    return_counts, tolog = ellipsis2false(return_counts, tolog)    
    bin_columns= is_iterable( bin_columns, exclude_string= True, 
                                 transform =True )
    tname = str(tname) ; #bin_column = str(bin_column)
    target_all_counts =[]
    
    existfeatures(data, features =bin_columns + [tname] )
    d= data.copy() 
    # -convert all features dtype to float for consistency
    # except the binary target 
    feature_cols = is_in_if (d.columns , tname, return_diff= True ) 
    d[feature_cols] = d[feature_cols].astype ( float)
    # -------------------------------------------------
    for bin_column in bin_columns: 
        d, tc  = _single_counts(d , bin_column, tname, 
                           odds =odds, 
                           tolog=tolog, 
                           return_counts= return_counts
                           )
    
        target_all_counts.append (tc) 
    # lowering the computation time 
    if return_counts: 
        d = ( target_all_counts if len(target_all_counts) >1 
                 else target_all_counts [0]
                 ) 

    return d

def _single_counts ( 
        d,/,  bin_column, tname, odds = "N+",
        tolog= False, return_counts = False ): 
    """ An isolated part of bin counting. 
    Compute single bin_counting. """
    # polish pos_label 
    od = copy.deepcopy( odds) 
    # reconvert log and removetrailer
    odds= str(odds).upper().replace ("_", "")
    # just separted for 
    keys = ('N-', 'N+', 'lOGN+')
    msg = ("Odds ratio or log Odds ratio expects"
           f" {smart_format(('N-', 'N+', 'logN+'), 'or')}. Got {od!r}")
    # check wther log is included 
    if odds.find('LOG')>=0: 
        tolog=True # then remove log 
        odds= odds.replace ("LOG", "")

    if odds not in keys: 
        raise ValueError (msg) 
    # If tolog, then reconstructs
    # the odds_labels
    if tolog: 
        odds= f"log{odds}"
    
    target_counts= _target_counting(
        d.filter(items=[bin_column, tname]),
    bin_column , tname =tname, 
    )
    target_all, target_bin_counts = _bin_counting(target_counts, tname, odds)
    # Check to make sure we have all the devices
    target_all.sort_values(by = f'total_{tname}', ascending=False)  
    if return_counts: 
        return d, target_all 
   
    # zip index with ratio 
    lr = list(zip (target_bin_counts.index, target_bin_counts[odds])
         )
    ybin = np.array ( d[bin_column])# replace value with ratio 
    for (value , ratio) in lr : 
        ybin [ybin ==value] = ratio 
        
    d[bin_column] = ybin 
    
    return d, target_all

def _target_counting(d, / ,  bin_column, tname ):
    """ An isolated part of counting the target. 
    
    :param d: DataFrame 
    :param bin_column: str, columns to appling bincounting strategy 
    :param tname: str, target name. 

    """
    pos_action = pd.Series(d[d[tname] > 0][bin_column].value_counts(),
        name=tname)
    
    neg_action = pd.Series(d[d[tname] < 1][bin_column].value_counts(),
    name=f'no_{tname}')
     
    counts = pd.DataFrame([pos_action,neg_action])._T.fillna('0')
    counts[f'total_{tname}'] = counts[tname].astype('int64') +\
    counts[f'no_{tname}'].astype('int64')
    
    return counts

def _bin_counting (counts, tname, odds="N+" ):
    """ Bin counting application to the target. 
    :param counts: pd.Series. Target counts 
    :param tname: str target name. 
    :param odds: str, label to bin-compute
    """
    counts['N+'] = ( counts[tname]
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64')
                            )
                    )
    counts['N-'] = ( counts[f'no_{tname}']
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64'))
                    )
    
    items2filter= ['N+', 'N-']
    if str(odds).find ('log')>=0: 
        counts['logN+'] = counts['N+'].divide(counts['N-'])
        counts ['logN-'] = counts ['N-'].divide ( counts['N+'])
        items2filter.extend (['logN+', 'logN-'])
    # If we wanted to only return bin-counting properties, 
    # we would filter here
    bin_counts = counts.filter(items= items2filter)

    return counts, bin_counts  
 
def laplace_smoothing_word(word, class_, /, word_counts, class_counts, V):
    """
    Apply Laplace smoothing to estimate the conditional probability of a 
    word given a class.

    Laplace smoothing (add-one smoothing) is used to handle the issue of 
    zero probability in categorical data, particularly in the context of 
    text classification with Naive Bayes.

    The mathematical formula for Laplace smoothing is:
    
    .. math:: 
        P(w|c) = \frac{\text{count}(w, c) + 1}{\text{count}(c) + |V|}

    where `count(w, c)` is the count of word `w` in class `c`, `count(c)` is 
    the total count of all words in class `c`, and `|V|` is the size of the 
    vocabulary.

    Parameters
    ----------
    word : str
        The word for which the probability is to be computed.
    class_ : str
        The class for which the probability is to be computed.
    word_counts : dict
        A dictionary containing word counts for each class. The keys should 
        be tuples of the form (word, class).
    class_counts : dict
        A dictionary containing the total count of words for each class.
    V : int
        The size of the vocabulary, i.e., the number of unique words in 
        the dataset.

    Returns
    -------
    float
        The Laplace-smoothed probability of the word given the class.

    Example
    -------
    >>> word_counts = {('dog', 'animal'): 3, ('cat', 'animal'):
                       2, ('car', 'non-animal'): 4}
    >>> class_counts = {'animal': 5, 'non-animal': 4}
    >>> V = len(set([w for (w, c) in word_counts.keys()]))
    >>> laplace_smoothing('dog', 'animal', word_counts, class_counts, V)
    0.4444444444444444
    
    References
    ----------
    - C.D. Manning, P. Raghavan, and H. Schütze, "Introduction to Information Retrieval",
      Cambridge University Press, 2008.
    - A detailed explanation of Laplace Smoothing can be found in Chapter 13 of 
      "Introduction to Information Retrieval" by Manning et al.

    Notes
    -----
    This function is particularly useful in text classification tasks where the
    dataset may contain a large number of unique words, and some words may not 
    appear in the training data for every class.
    """
    word_class_count = word_counts.get((word, class_), 0)
    class_word_count = class_counts.get(class_, 0)
    probability = (word_class_count + 1) / (class_word_count + V)
    return probability

def laplace_smoothing_categorical(
        data, /, feature_col, class_col, V=None):
    """
    Apply Laplace smoothing to estimate conditional probabilities of 
    categorical features given a class in a dataset.

    This function calculates the Laplace-smoothed probabilities for each 
    category of a specified feature given each class.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing categorical features and a class label.
    feature_col : str
        The column name in the dataset representing the feature for which 
        probabilities are to be calculated.
    class_col : str
        The column name in the dataset representing the class label.
    V : int or None, optional
        The size of the vocabulary (i.e., the number of unique categories 
                                    in the feature).
        If None, it will be calculated based on the provided feature column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Laplace-smoothed probabilities for each 
        category of the feature across each class.

    Example
    -------
    >>> data = pd.DataFrame({'feature': ['cat', 'dog', 'cat', 'bird'],
                             'class': ['A', 'A', 'B', 'B']})
    >>> probabilities = laplace_smoothing_categorical(data, 'feature', 'class')
    >>> print(probabilities)

    Notes
    -----
    This function is useful for handling categorical data in classification
    tasks, especially when the dataset may contain categories that do not 
    appear in the training data for every class.
    """
    if V is None:
        V = data[feature_col].nunique()

    class_counts = data[class_col].value_counts()
    probability_table = pd.DataFrame()

    # Iterating over each class to calculate probabilities
    for class_value in data[class_col].unique():
        class_subset = data[data[class_col] == class_value]
        feature_counts = class_subset[feature_col].value_counts()
        probabilities = (feature_counts + 1) / (class_counts[class_value] + V)
        probabilities.name = class_value
        probability_table = probability_table.append(probabilities)

    return probability_table.fillna(1 / V)

def laplace_smoothing(
    data, /,  
    alpha=1,
    columns=None, 
    as_frame=False, 
    ):
    """
    Apply Laplace Smoothing to a dataset.

    Parameters
    ----------
    data : ndarray
        An array-like object containing categorical data. Each column 
        represents a feature, and each row represents a data sample.
    alpha : float, optional
        The smoothing parameter, often referred to as 'alpha'. This is 
        added to the count for each category in each feature. 
        Default is 1 (Laplace Smoothing).
    
    columns: list, 
       Columns to construct the data. 
    as_frame: bool, default=False, 
       To convert data as a frame before proceeding. 
       
    Returns
    -------
    smoothed_probs : ndarray
        An array of the same shape as `data` containing the smoothed 
        probabilities for each category in each feature.

    References
    ----------
    - C.D. Manning, P. Raghavan, and H. Schütze, "Introduction to Information Retrieval",
      Cambridge University Press, 2008.
    - A detailed explanation of Laplace Smoothing can be found in Chapter 13 of 
      "Introduction to Information Retrieval" by Manning et al.
      
    Notes
    -----
    This implementation assumes that the input data is categorical 
    and encoded as non-negative integers, which are indices of categories.

    Examples
    --------
    >>> data = np.array([[0, 1], [1, 0], [1, 1]])
    >>> laplace_smoothing(data, alpha=1)
    array([[0.4 , 0.6 ],
           [0.6 , 0.4 ],
           [0.6 , 0.6 ]])
    """
    data = build_data_if(data, columns= columns, to_frame =as_frame ) 
    # Count the occurrences of each category in each feature
    n_samples, n_features = data.shape
    feature_counts = [np.bincount(data[:, i], minlength=np.max(data[:, i]) + 1)
                      for i in range(n_features)]

    # Apply Laplace smoothing
    smoothed_counts = [counts + alpha for counts in feature_counts]
    total_counts = [counts.sum() for counts in smoothed_counts]

    # Calculate probabilities
    smoothed_probs = np.array([counts / total for counts, total in
                               zip(smoothed_counts, total_counts)])
    
    # Transpose and return the probabilities corresponding to each data point
    return smoothed_probs._T[data]
  
def evaluate_model(
    model: _F, 
    X:NDArray |DataFrame, 
    y: ArrayLike |Series, 
    Xt:NDArray |DataFrame, 
    yt:ArrayLike |Series=None, 
    scorer:str | _F = 'accuracy',
    eval:bool =False,
    **kws
    ): 
    """ Evaluate model and quick test the score with metric scorers. 
    
    Parameters
    --------------
    model: Callable, {'preprocessor + estimator } | estimator,
        the preprocessor is list of step for data handling all encapsulated 
        on the pipeline. model can also be a simple estimator with `fit`,
        
    X: N-d array, shape (N, M) 
       the training set composed of N-columns and the M-samples. The 
        feature set excludes the target `y`. 
    y: arraylike , shape (M)
        the target is composed of M-examples in supervised learning. 
    
    Xt: N-d array, shape (N, M) 
        test set array composed of N-columns and the M-samples. The 
        feature set excludes the target `y`. 
    yt: arraylike , shape (M)
        test label (or test target)  composed of M-examples in 
        supervised learning.
        
    scorer: str, Callable, 
        a scorer is a metric  function for model evaluation. If given as string 
        it should be the prefix of the following metrics: 
            
            * "classification_report"     -> for classification_report,
            * 'precision_recall'          -> for precision_recall_curve,
            * "confusion_matrix"          -> for a confusion_matrix,
            * 'precision'                 -> for  precision_score,
            * "accuracy"                  -> for  accuracy_score
            * "mse"                       -> for mean_squared_error, 
            * "recall"                    -> for  recall_score, 
            * 'auc'                       -> for  roc_auc_score, 
            * 'roc'                       -> for  roc_curve 
            * 'f1'                        -> for f1_score,
            
        Other string prefix values should raises an errors 
        
    kws: dict, 
        Additionnal keywords arguments from scklearn metric function.
        
    Returns 
    ----------
    Tuple : (score, ypred)
        the model score or the predicted y if `predict` is set to ``True``. 
        
    """
    score = None 
    if X.ndim ==1: 
        X = X.reshape(-1, 1) 
    if Xt.ndim ==1: 
        Xt = Xt.reshape(-1, 1)
        
    model.fit(X, y)
    # model.transform(X, y)
    ypred = model.predict(Xt)
    
    if eval : 
        if yt is None: 
            raise TypeError(" NoneType 'yt' cannot be used for model evaluation.")
            
        if scorer is None: 
           scorer =  _scorers['accuracy']
           
        if isinstance (scorer, str): 
            if str(scorer) not in _scorers.keys(): 
                raise ValueError (
                    "Given scorer {scorer!r }is unknown. Accepts "
                    f" only {smart_format(_scorers.keys())}") 
                
            scorer = _scorers.get(scorer)
        elif not hasattr (scorer, '__call__'): 
            raise TypeError ("scorer should be a callable object,"
                             f" got {type(scorer).__name__!r}")
            
        score = scorer (yt, ypred, **kws)
    
    return  ypred, score  

def get_correlated_features(
        df:DataFrame ,
        corr:str ='pearson', 
        threshold: float=.95 , 
        fmt: bool= False 
        )-> DataFrame: 
    """Find the correlated features/columns in the dataframe. 
    
    Indeed, highly correlated columns don't add value and can throw off 
    features importance and interpretation of regression coefficients. If we  
    had correlated columns, choose to remove either the columns from  
    level_0 or level_1 from the features data is a good choice. 
    
    Parameters 
    -----------
    df: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
        Dataframe containing samples M  and features N
    corr: str, ['pearson'|'spearman'|'covariance']
        Method of correlation to perform. Note that the 'person' and 
        'covariance' don't support string value. If such kind of data 
        is given, turn the `corr` to `spearman`. *default* is ``pearson``
        
    threshold: int, default is ``0.95``
        the value from which can be considered as a correlated data. Should not 
        be greater than 1. 
        
    fmt: bool, default {``False``}
        format the correlated dataframe values 
        
    Returns 
    ---------
    df: `pandas.DataFrame`
        Dataframe with cilumns equals to [level_0, level_1, pearson]
        
    Examples
    --------
    >>> from gofast.tools.mlutils import get_correlated_features 
    >>> df_corr = get_correlated_features (data , corr='spearman',
                                     fmt=None, threshold=.95
                                     )
    """
    th= copy.deepcopy(threshold) 
    threshold = str(threshold)  
    try : 
        threshold = float(threshold.replace('%', '')
                          )/1e2  if '%' in threshold else float(threshold)
    except: 
        raise TypeError (
            f"Threshold should be a float value, got: {type(th).__name__!r}")
          
    if threshold >= 1 or threshold <= 0 : 
        raise ValueError (
            f"threshold must be ranged between 0 and 1, got {th!r}")
      
    if corr not in ('pearson', 'covariance', 'spearman'): 
        raise ValueError (
            f"Expect ['pearson'|'spearman'|'covariance'], got{corr!r} ")
    # collect numerical values and exclude cat values
    
    df = select_features(df, include ='number')
        
    # use pipe to chain different func applied to df 
    c_df = ( 
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril (df1, k=-1 ), # low triangle zeroed 
                columns = df.columns, 
                index =df.columns, 
                )
            )
            .stack ()
            .rename(corr)
            .pipe(
                lambda s: s[
                    s.abs()> threshold 
                    ].reset_index()
                )
                .query("level_0 not in level_1")
        )

    return  c_df.style.format({corr :"{:2.f}"}) if fmt else c_df 
                      
def get_target (df, tname, inplace = True): 
    """ Extract target and modified data in place or not . 
    
    :param df: A dataframe with features including the target name `tname`
    :param tname: A target name. It should be include in the dataframe columns 
        otherwise an error is raised. 
    :param inplace: modified the dataframe inplace. if ``False`` return the 
        dataframe. the *defaut* is ``True`` 
        
    :returns: Tuple of the target and dataframe (modified or not)
    
    :example: 
    >>> from gofast.datasets import fetch_data '
    >>> from gofast.tools.mlutils import exporttarget 
    >>> data0 = fetch_data ('bagoue original').get('data=dfy1') 
    >>> # no modification 
    >>> target, data_no = exporttarget (data0 , 'sfi', False )
    >>> len(data_no.columns ) , len(data0.columns ) 
    ... (13, 13)
    >>> # modified in place 
    >>> target, data= exporttarget (data0 , 'sfi')
    >>> len(data.columns ) , len(data0.columns ) 
    ... (12, 12)
        
    """
    df = _assert_all_types(df, pd.DataFrame)
    existfeatures(df, tname) # assert tname 
    if is_iterable(tname, exclude_string=True): 
        tname = list(tname)
        
    t = df [tname ] 
    df.drop (tname, axis =1 , inplace =inplace )
    
    return t, df

def features_in (data, / ,  features, error ='raise'): 
    """ Control whether the feature exists in the data
    
    :param data: dict, 
    """ 
    return existfeatures(build_data_if(data), features, error = error )

def existfeatures (df, features, error='raise'): 
    """Control whether the features exist or not  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"Feature{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'is' if nfeat <2 else 'are'}"
                                  " missing in the data attributes.")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf  
    
def select_features(
    data: DataFrame,
    features: List[str] =None, 
    include = None, 
    exclude = None,
    coerce: bool=...,
    columns: list=None, 
    verify_integrity:bool=..., 
	parse_features: bool=..., 
    **kwd
    ): 
    """ Select features  and return new dataframe.  
    
    :param data: a dataframe for features selections 
    :param features: list of features to select. List of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieve in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param columns: list, needs columns to construst a dataframe if data is 
        passed as Numpy object array.
    :param verify_integrity: bool, Control the data type and rebuilt the data 
       to the right type.
    :param parse_features:bool, parse the string and convert to an iterable object.
    :param kwd: additional keywords arguments from `pd.astype` function 
    
    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    
    :examples: 
        >>> from gofast.tools.mlutils import select_features 
        >>> data = {"Color": ['Blue', 'Red', 'Green'], 
                    "Name": ['Mary', "Daniel", "Augustine"], 
                    "Price ($)": ['200', "300", "100"]
                    }
        >>> select_features (data, include='number')
        Out[230]: 
        Empty DataFrame
        Columns: []
        Index: [0, 1, 2]
        >>> select_features (data, include='number', verify_integrity =True )
        Out[232]: 
            Price ($)
        0       200.0
        1       300.0
        2       100.0
        >>> select_features (data, features =['Color', 'Price ($)'], )
        Out[234]: 
           color  Price ($)
        0   Blue        200
        1    Red        300
        2  Green        100
    """
    coerce, verify_integrity, parse_features= ellipsis2false( 
        coerce, verify_integrity, parse_features)
    
    data = build_data_if(data, columns = columns, )
  
    if verify_integrity: 
        data = to_numeric_dtypes(data )
        
    if features is not None: 
        features= list(is_iterable (
            features, exclude_string=True, transform=True, 
            parse_string = parse_features)
            )
        existfeatures(data, features, error ='raise')
    # change the dataype 
    data = data.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return data [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return data if coerce else data.select_dtypes (include, exclude) 
    
def get_global_score(
    cvres: Dict[str, ArrayLike],
    ignore_convergence_problem: bool = False
) -> Tuple[float, float]:
    """
    Retrieve the global mean and standard deviation of test scores from 
    cross-validation results.

    This function computes the overall mean and standard deviation of test 
    scores from the results of cross-validation. It can also handle situations 
    where convergence issues might have occurred during model training, 
    depending on the `ignore_convergence_problem` flag.

    Parameters
    ----------
    cvres : Dict[str, np.ndarray]
        A dictionary containing the cross-validation results. Expected to have 
        keys 'mean_test_score' and 'std_test_score', with each key mapping to 
        an array of scores.
    ignore_convergence_problem : bool, default=False
        If True, ignores NaN values that might have resulted from convergence 
        issues during model training while calculating the mean. If False, NaN 
        values contribute to the final mean as NaN.

    Returns
    -------
    Tuple[float, float]
        A tuple containing two float values:
        - The first element is the mean of the test scores across all 
          cross-validation folds.
        - The second element is the mean of the standard deviations of the 
          test scores across all cross-validation folds.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> clf = DecisionTreeClassifier()
    >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5,
    ...                          scoring='accuracy', return_train_score=True)
    >>> cvres = {'mean_test_score': scores, 'std_test_score': np.std(scores)}
    >>> mean_score, mean_std = get_global_score(cvres)
    >>> print(f"Mean score: {mean_score}, Mean standard deviation: {mean_std}")

    Notes
    -----
    - The function is primarily designed to be used with results obtained from 
      scikit-learn's cross-validation functions like `cross_val_score`.
    - It is assumed that `cvres` contains keys 'mean_test_score' and 
      'std_test_score'.
    """
    if ignore_convergence_problem:
        mean_score = np.nanmean(cvres.get('mean_test_score'))
        mean_std = np.nanmean(cvres.get('std_test_score'))
    else:
        mean_score = cvres.get('mean_test_score').mean()
        mean_std = cvres.get('std_test_score').mean()

    return mean_score, mean_std

def cfexist(features_to: List[ArrayLike], 
            features: List[str] )-> bool:      
    """
    Control features existence into another list . List or array can be a 
    dataframe columns for pratical examples.  
    
    :param features_to :list of array to be controlled .
    :param features: list of whole features located on array of `pd.DataFrame.columns` 
    
    :returns: 
        -``True``:If the provided list exist in the features colnames 
        - ``False``: if not 

    """
    if isinstance(features_to, str): 
        features_to =[features_to]
    if isinstance(features, str): features =[features]
    
    if sorted(list(features_to))== sorted(list(
            set(features_to).intersection(set(features)))): 
        return True
    else: return False 

def formatGenericObj(generic_obj :Iterable[_T])-> _T: 
    """
    Format a generic object using the number of composed items. 

    :param generic_obj: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    
    :Example: 
        
        >>> from gofast.tools.mlutils import formatGenericObj 
        >>> formatGenericObj ({'ohmS', 'lwi', 'power', 'id', 
        ...                         'sfi', 'magnitude'})
        
    """
    
    return ['{0}{1}{2}'.format('{', ii, '}') for ii in range(
                    len(generic_obj))]

def find_relation_between_generics(
    gen_obj1: Iterable[Any],
    gen_obj2: Iterable[Any],
    operation: str = "intersection"
) -> Set[Any]:
    """
    Computes either the intersection or difference of two generic iterable objects.

    Based on the specified operation, this function finds either common elements 
    (intersection) or unique elements (difference) between two iterable objects 
    like lists, sets, or dictionaries.

    Parameters
    ----------
    gen_obj1 : Iterable[Any]
        The first generic iterable object. Can be a list, set, dictionary, 
        or any iterable type.
    gen_obj2 : Iterable[Any]
        The second generic iterable object. Same as gen_obj1.
    operation : str, optional
        The operation to perform. Can be 'intersection' or 'difference'.
        Defaults to 'intersection'.

    Returns
    -------
    Set[Any]
        A set containing either the common elements (intersection) or 
        unique elements (difference) of the two iterables.

    Examples
    --------
    Intersection:
    >>> from gofast.tools.mlutils import find_relation_between_generics
    >>> result = find_relation_between_generics(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    >>> print(result)
    {'ohmS', 'lwi', 'power'}

    Difference:
    >>> result = find_relation_between_generics(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'},
    ...     operation='difference'
    ... )
    >>> print(result)
    {'id', 'sfi', 'magnitude'}

    Notes
    -----
    The function returns the result as a set, irrespective of the
    type of the input iterables. The 'operation' parameter controls
    whether the function calculates the intersection or difference.
    """

    set1 = set(gen_obj1)
    set2 = set(gen_obj2)

    if operation == "intersection":
        return set1.intersection(set2)
    elif operation == "difference":
        if len(gen_obj1) <= len(gen_obj2):
            return set(gen_obj2).difference(set(gen_obj1))
        else:
            return set(gen_obj1).difference(set(gen_obj2))
    else:
        raise ValueError("Invalid operation specified. Choose"
                         " 'intersection' or 'difference'.")

def find_intersection_between_generics(
    gen_obj1: Iterable[Any],
    gen_obj2: Iterable[Any]
) -> Set[Any]:
    """
    Computes the intersection of two generic iterable objects.

    This function finds common elements between two iterable objects 
    (like lists, sets, or dictionaries) and returns a set containing 
    these shared elements. The function is designed to handle various 
    iterable types.

    Parameters
    ----------
    gen_obj1 : Iterable[Any]
        The first generic iterable object. Can be a list, set, dictionary, 
        or any iterable type.
    gen_obj2 : Iterable[Any]
        The second generic iterable object. Same as gen_obj1.

    Returns
    -------
    Set[Any]
        A set containing the elements common to both iterables.

    Example
    -------
    >>> from gofast.tools.mlutils import find_intersection_between_generics
    >>> result = find_intersection_between_generics(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    >>> print(result)
    {'ohmS', 'lwi', 'power'}

    Notes
    -----
    The function returns the intersection as a set, irrespective of the
    type of the input iterables.
    """

    # Convert both iterables to sets for intersection calculation
    set1 = set(gen_obj1)
    set2 = set(gen_obj2)

    # Calculate and return the intersection
    return set1.intersection(set2)

def findIntersectionGenObject(
        gen_obj1: Iterable[Any], 
        gen_obj2: Iterable[Any]
                              )-> set: 
    """
    Find the intersection of generic object and keep the shortest len 
    object `type` at the be beginning 
  
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from gofast.tools.mlutils import findIntersectionGenObject
        >>> findIntersectionGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) <= len(gen_obj2):
        objType = type(gen_obj1)
    else: objType = type(gen_obj2)

    return objType(set(gen_obj1).intersection(set(gen_obj2)))

def find_difference_between_generics(
    gen_obj1: Iterable[Any],
    gen_obj2: Iterable[Any]
   ) -> Union[None, Set[Any]]:
    """
    Identifies the difference between two generic iterable objects.

    This function computes the difference between two iterable objects 
    (like lists or sets) and returns a set containing elements that are 
    unique to the larger iterable. If both iterables are of the same length, 
    the function returns None.

    Parameters
    ----------
    gen_obj1 : Iterable[Any]
        The first generic iterable object. Can be a list, set, dictionary, 
        or any iterable type.
    gen_obj2 : Iterable[Any]
        The second generic iterable object. Same as gen_obj1.

    Returns
    -------
    Union[None, Set[Any]]
        A set containing the unique elements from the larger iterable.
        Returns None if both
        iterables are of equal length.

    Example
    -------
    >>> from gofast.tools.mlutils import find_difference_between_generics
    >>> result = find_difference_between_generics(
    ...     ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'],
    ...     {'ohmS', 'lwi', 'power'}
    ... )
    >>> print(result)
    {'id', 'sfi', 'magnitude'}
    """

    # Convert both iterables to sets for difference calculation
    set1 = set(gen_obj1)
    set2 = set(gen_obj2)

    # Calculate difference based on length
    if len(set1) > len(set2):
        return set1.difference(set2)
    elif len(set1) < len(set2):
        return set2.difference(set1)

    # Return None if both are of equal length
    return None

def findDifferenceGenObject(gen_obj1: Iterable[Any],
                            gen_obj2: Iterable[Any]
                              )-> None | set: 
    """
    Find the difference of generic object and keep the shortest len 
    object `type` at the be beginning: 
 
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from gofast.tools.mlutils import findDifferenceGenObject
        >>> findDifferenceGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) < len(gen_obj2):
        objType = type(gen_obj1)
        return objType(set(gen_obj2).difference(set(gen_obj1)))
    elif len(gen_obj1) > len(gen_obj2):
        objType = type(gen_obj2)
        return objType(set(gen_obj1).difference(set(gen_obj2)))
    else: return 
   
    return set(gen_obj1).difference(set(gen_obj2))
    
def featureExistError(superv_features: Iterable[_T], 
                      features:Iterable[_T]) -> None:
    """
    Catching feature existence errors.
    
    check error. If nothing occurs  then pass 
    
    :param superv_features: 
        list of features presuming to be controlled or supervised
        
    :param features: 
        List of all features composed of pd.core.DataFrame. 
    
    """
    for ii, supff in enumerate([superv_features, features ]): 
        if isinstance(supff, str): 
            if ii==0 : superv_features=[superv_features]
            if ii==1 :features =[superv_features]
            
    try : 
        resH= cfexist(features_to= superv_features,
                           features = features)
    except TypeError: 
        
        print(' Features can not be a NoneType value.'
              'Please set a right features.')
        _logger.error('NoneType can not be a features!')
    except :
        raise ParameterNumberError  (
           f'Parameters number of {features} is  not found in the '
           ' dataframe columns ={0}'.format(list(features)))
    
    else: 
        if not resH:  raise ParameterNumberError  (
            f'Parameters number is ``{features}``. NoneType object is'
            ' not allowed in  dataframe columns ={0}'.
            format(list(features)))
        
def control_existing_estimator(
    estimator_name: str, 
    raise_error: bool = False
) -> Union[Tuple[str, str], None]:
    """
    Validates and retrieves the corresponding prefix for a given estimator name.

    This function checks if the provided estimator name exists in a predefined
    list of estimators. If found, it returns the corresponding prefix and full name.
    Otherwise, it either raises an error or returns None, based on the 
    'raise_error' flag.

    Parameters
    ----------
    estimator_name : str
        The name of the estimator to check.
    raise_error : bool, default False
        If True, raises an error when the estimator is not found. Otherwise, 
        emits a warning.

    Returns
    -------
    Tuple[str, str] or None
        A tuple containing the prefix and full name of the estimator, or 
        None if not found.

    Example
    -------
    >>> from gofast.tools.mlutils import control_existing_estimator
    >>> test_est = control_existing_estimator('svm')
    >>> print(test_est)
    ('svc', 'SupportVectorClassifier')
    """

    estimator_name = estimator_name.lower().strip()
    for prefix, names in _estimators.items():
        lower_names = [name.lower() for name in names]
        
        if estimator_name in lower_names:
            return prefix, names[0]

    if raise_error:
        valid_names = [name for names in _estimators.values() for name in names]
        raise EstimatorError(f'Unsupported estimator {estimator_name!r}. '
                             f'Expected one of {valid_names}.')
    else:
        available_estimators = [name for names in _estimators.values() 
                                for name in names]
        warning_msg = (f"Estimator {estimator_name!r} not found. "
                       f"Expected one of: {available_estimators}.")
        warnings.warn(warning_msg)

    return None
       
def controlExistingEstimator(
        estimator_name: str , raise_err =False ) -> Union [Dict[str, _T], None]: 
    """ 
    When estimator name is provided by user , will chech the prefix 
    corresponding

    Catching estimator name and find the corresponding prefix 
        
    :param estimator_name: Name of given estimator 
    
    :Example: 
        
        >>> from gofast.tools.mlutils import controlExistingEstimator 
        >>> test_est =controlExistingEstimator('svm')
        ('svc', 'SupportVectorClassifier')
        
    """
    estimator_name = str(estimator_name).lower().strip() 
    e = None ; efx = None 
    for k, v in _estimators.items() : 
        v_ = list(map(lambda o: str(o).lower(), v)) 
        
        if estimator_name in v_ : 
            e, efx = k, v[0]
            break 

    if e is None: 
        ef = map(lambda o: o[0], _estimators.values() )
        if raise_err: 
            raise EstimatorError(f'Unsupport estimator {estimator_name!r}.'
                                 f' Expect {smart_format(ef)}') 
        ef =list(ef)
        emsg = f"Default estimator {estimator_name!r} not found!" +\
            (" Expect: {}".format(formatGenericObj(ef)
                                  ).format(*ef))

        warnings.warn(emsg)
        
            
        return 
    
    return e, efx 

def format_model_score(
    model_score: Union[float, Dict[str, float]] = None,
    selected_estimator: Optional[str] = None
) -> None:
    """
    Formats and prints model scores.

    Parameters
    ----------
    model_score : float or Dict[str, float], optional
        The model score or a dictionary of model scores with estimator 
        names as keys.
    selected_estimator : str, optional
        Name of the estimator to format the score for. Used only if 
        `model_score` is a float.

    Example
    -------
    >>> from gofast.tools.mlutils import format_model_score
    >>> format_model_score({'DecisionTreeClassifier': 0.26, 'BaggingClassifier': 0.13})
    >>> format_model_score(0.75, selected_estimator='RandomForestClassifier')
    """

    print('-' * 77)
    if isinstance(model_score, dict):
        for estimator, score in model_score.items():
            formatted_score = round(score * 100, 3)
            print(f'> {estimator:<30}:{"Score":^10}= {formatted_score:^10} %')
    elif isinstance(model_score, float):
        estimator_name = selected_estimator if selected_estimator else 'Unknown Estimator'
        formatted_score = round(model_score * 100, 3)
        print(f'> {estimator_name:<30}:{"Score":^10}= {formatted_score:^10} %')
    else:
        print('Invalid model score format. Please provide a float or'
              ' a dictionary of scores.')
    print('-' * 77)
    
def formatModelScore(
        model_score: Union [float, Dict[str, float]] = None,
        select_estimator: str = None ) -> None   : 
    """
    Format the result of `model_score`
        
    :param model_score: Can be float or dict of float where key is 
                        the estimator name 
    :param select_estimator: Estimator name 
    
    :Example: 
        
        >>> from gofast.tools.mlutils import formatModelScore 
        >>>  formatModelScore({'DecisionTreeClassifier':0.26, 
                      'BaggingClassifier':0.13}
        )
    """ 
    print('-'*77)
    if isinstance(model_score, dict): 
        for key, val in model_score.items(): 
            print('> {0:<30}:{1:^10}= {2:^10} %'.format( key,' Score', round(
                val *100,3 )))
    else : 
        if select_estimator is None : 
            select_estimator ='___'
        if inspect.isclass(select_estimator): 
            select_estimator =select_estimator.__class__.__name__
        
        try : 
            _, select_estimator = controlExistingEstimator(select_estimator)
        
        except : 
            if select_estimator is None :
                select_estimator =str(select_estimator)
            else: select_estimator = '___'
            
        print('> {0:<30}:{1:^10}= {2:^10} %'.format(select_estimator,
                     ' Score', round(
            model_score *100,3 )))
        
    print('-'*77)
    

def stats_from_prediction(y_true, y_pred, verbose=False):
    """
    Generate statistical summaries and accuracy metrics from actual values (y_true)
    and predicted values (y_pred).

    Parameters
    ----------
    y_true : list or numpy.array
        Actual values.
    y_pred : list or numpy.array
        Predicted values.
    verbose : bool, optional
        If True, print the statistical summary and accuracy metrics.
        Default is False.

    Returns
    -------
    dict
        A dictionary containing statistical measures such 
        as MAE, MSE, RMSE, 
        and accuracy (if applicable).

    Examples
    --------
    >>> from gofast.tools.mlutils import stats_from_prediction 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> stats_from_prediction(y_true, y_pred, verbose=True)
    """
    # Calculating statistics
    check_consistent_length(y_true, y_pred )
    stats = {
        'mean': np.mean(y_pred),
        'median': np.median(y_pred),
        'std_dev': np.std(y_pred),
        'min': np.min(y_pred),
        'max': np.max(y_pred)
    }
    # add the metric stats 
    stats =dict ({
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        }, **stats, 
        )

    # Adding accuracy for classification tasks
    # Check if y_true and y_pred are categories task 
    if is_classification_task(y_true, y_pred ): 
    # if all(map(lambda x: x in [0, 1], y_true + y_pred)): #binary 
        stats['Accuracy'] = accuracy_score(y_true, y_pred)

    # Printing the results if verbose is True
    if verbose:
        fancy_printer(stats, "Prediction Statistics Summary" )

    return stats


def write_excel(
        listOfDfs: List[DataFrame],
        csv: bool =False , 
        sep:str =',') -> None: 
    """ 
    Rewrite excell workbook with dataframe for :ref:`read_from_excelsheets`. 
    
    Its recover the name of the files and write the data from dataframe 
    associated with the name of the `erp_file`. 
    
    :param listOfDfs: list composed of `erp_file` name at index 0 and the
     remains dataframes. 
    :param csv: output workbook in 'csv' format. If ``False`` will return un 
     `excel` format. 
    :param sep: type of data separation. 'default is ``,``.'
    
    """
    site_name = listOfDfs[0]
    listOfDfs = listOfDfs[1:]
    for ii , df in enumerate(listOfDfs):
        
        if csv:
            df.to_csv(df, sep=sep)
        else :
            with pd.ExcelWriter(f"z{site_name}_{ii}.xlsx") as writer: 
                df.to_excel(writer, index=False)
    

def fetch_tgz (
    data_url:str ,
    data_path:str ,
    tgz_filename:str 
   ) -> None: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)

    tgz_path = os.path.join(data_url, tgz_filename.replace('/', ''))
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path = data_path )
    data_tgz.close()
    
def fetch_tgz_from_url (
    data_url:str , 
    data_path:str ,
    tgz_file, 
    file_to_retreive=None,
    **kws
    ) -> Union [str, None]: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    
    :example: 
    >>> from gofast.tools.mlutils import fetch_tgz_from_url
    >>> DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'
    >>> # from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
    >>> DATA_PATH = 'data/__tar.tgz'  # 'BagoueCIV__dataset__main/__tar.tgz_files__'
    >>> TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
    >>> CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
    >>> fetch_tgz_from_url (data_url= DATA_URL,data_path=DATA_PATH,
                            tgz_filename=TGZ_FILENAME
                            ) 
    """
    f= None
    if data_url is not None: 
        tgz_path = os.path.join(data_path, tgz_file.replace('/', ''))
        try: 
            urllib.request.urlretrieve(data_url, tgz_path)
        except urllib.URLError: 
            print("<urlopen error [WinError 10061] No connection could "
                  "be made because the target machine actively refused it>")
        except ConnectionError or ConnectionRefusedError: 
            print("Connection failed!")
        except: 
            print(f"Unable to fetch {os.path.basename(tgz_file)!r}"
                  f" from <{data_url}>")
            
        return False 
    
    if file_to_retreive is not None: 
        f= fetch_tgz_locally(filename=file_to_retreive, **kws)
        
    return f

def fetch_tgz_locally(tgz_file: str , filename: str ,
        savefile: str ='tgz',rename_outfile: Optional [str]=None 
        ) -> str :
    """ Fetch single file from archived tar file and rename a file if possible.
    
    :param tgz_file: str or Path-Like obj 
        Full path to tarfile. 
    :param filename:str 
        Tagert  file to fetch from the tarfile.
    :savefile:str or Parh-like obj 
        Destination path to save the retreived file. 
    :param rename_outfile:str or Path-like obj
        Name of of the new file to replace the fetched file.
    :return: Location of the fetched file
    :Example: 
        >>> from gofast.tools.mlutils import fetchSingleTGZData
        >>> fetch_tgz_locally('data/__tar.tgz/fmain.bagciv.data.tar.gz', 
                               rename_outfile='main.bagciv.data.csv')
    """
     # get the extension of the fetched file 
    fetch_ex = os.path.splitext(filename)[1]
    if not os.path.isdir(savefile):
        os.makedirs(savefile)
    
    def retreive_main_member (tarObj): 
        """ Retreive only the main member that contain the target filename."""
        for tarmem in tarObj.getmembers():
            if os.path.splitext(tarmem.name)[1]== fetch_ex: #'.csv': 
                return tarmem 
            
    if not os.path.isfile(tgz_file):
        raise FileNotFoundError(f"Source {tgz_file!r} is a wrong file.")
   
    with tarfile.open(tgz_file) as tar_ref:
        tar_ref.extractall(members=[retreive_main_member(tar_ref)])
        tar_name = [ name for name in tar_ref.getnames()
                    if name.find(filename)>=0 ][0]
        shutil.move(tar_name, savefile)
        # for consistency ,tree to check whether the tar info is 
        # different with the collapse file 
        if tar_name != savefile : 
            # print(os.path.join(os.getcwd(),os.path.dirname(tar_name)))
            _fol = tar_name.split('/')[0]
            shutil.rmtree(os.path.join(os.getcwd(),_fol))
        # now rename the file to the 
        if rename_outfile is not None: 
            os.rename(os.path.join(savefile, filename), 
                      os.path.join(savefile, rename_outfile))
        if rename_outfile is None: 
            rename_outfile =os.path.join(savefile, filename)
            
        print(f"---> {os.path.join(savefile, rename_outfile)!r} was "
              f" successfully decompressed from {os.path.basename(tgz_file)!r}"
              f"and saved to {savefile!r}")
        
    return os.path.join(savefile, rename_outfile)
    
def load_csv ( data: str = None, delimiter: str  =None ,**kws
        )-> DataFrame:
    """ Load csv file and convert to a frame. 
    
    :param data_path: path to data csv file 
    :param delimiter: str, item for data  delimitations. 
    :param kws: dict, additional keywords arguments passed 
        to :class:`pandas.read_csv`
    :return: pandas dataframe 
    
    """ 
    if not os.path.isfile(data): 
        raise TypeError("Expect a valid CSV file.")
    if (os.path.splitext(data)[1].replace('.', '')).lower() !='csv': 
        raise ValueError("Read only a csv file.")
        
    return pd.read_csv(data, delimiter=delimiter, **kws) 


def split_train_test (
        df:DataFrame[DType[_T]],
        test_ratio:float 
        )-> Tuple [DataFrame[DType[_T]]]: 
    """ A naive dataset split into train and test sets from a ratio and return 
    a shuffled train set and test set.
        
    :param df: a dataframe containing features 
    :param test_ratio: a ratio for test set batch. `test_ratio` is ranged 
        between 0 to 1. Default is 20%.
        
    :returns: a tuple of train set and test set. 
    
    """
    if isinstance (test_ratio, str):
        if test_ratio.lower().find('%')>=0: 
            try: test_ratio = float(test_ratio.lower().replace('%', ''))/100.
            except: TypeError (f"Could not convert value to float: {test_ratio!r}")
    if test_ratio <=0: 
        raise ValueError ("Invalid ratio. Must greater than 0.")
    elif test_ratio >=1: 
        raise ValueError("Invalid ratio. Must be less than 1 and greater than 0.")
        
    shuffled_indices =np.random.permutation(len(df)) 
    test_set_size = int(len(df)* test_ratio)
    test_indices = shuffled_indices [:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return df.iloc[train_indices], df.iloc[test_indices]
    
def test_set_check_id (
        identifier:int, 
        test_ratio: float , 
        hash:_F[_T]
        ) -> bool: 
    """ 
    Get the test set id and set the corresponding unique identifier. 
    
    Compute the a hash of each instance identifier, keep only the last byte 
    of the hash and put the instance in the testset if this value is lower 
    or equal to 51(~20% of 256) 
    has.digest()` contains object in size between 0 to 255 bytes.
    
    :param identifier: integer unique value 
    :param ratio: ratio to put in test set. Default is 20%. 
    
    :param hash:  
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.
    """
    return hash(np.int64(identifier)).digest()[-1]< 256 * test_ratio

def split_train_test_by_id(
    data:DataFrame,
    test_ratio:float,
    id_column:Optional[List[int]]=None,
    keep_colindex:bool=True, 
    hash : _F =hashlib.md5
    )-> Tuple[ _Sub[DataFrame[DType[_T]]], _Sub[DataFrame[DType[_T]]]] : 
    """
    Ensure that data will remain consistent accross multiple runs, even if 
    dataset is refreshed. 
    
    The new testset will contain 20%of the instance, but it will not contain 
    any instance that was previously in the training set.

    :param data: Pandas.core.DataFrame 
    :param test_ratio: ratio of data to put in testset 
    :param id_colum: identifier index columns. If `id_column` is None,  reset  
                dataframe `data` index and set `id_column` equal to ``index``
    :param hash: secures hashes algorithms. Refer to 
                :func:`~test_set_check_id`
    :returns: consistency trainset and testset 
    """
    if isinstance(data, np.ndarray) : 
        data = pd.DataFrame(data) 
        if 'index' in data.columns: 
            data.drop (columns='index', inplace=True)
            
    if id_column is None: 
        id_column ='index' 
        data = data.reset_index() # adds an `index` columns
        
    ids = data[id_column]
    in_test_set =ids.apply(lambda id_:test_set_check_id(id_, test_ratio, hash))
    if not keep_colindex: 
        data.drop (columns ='index', inplace =True )
        
    return data.loc[~in_test_set], data.loc[in_test_set]

def discretize_categories(
        data: Union [ArrayLike, DataFrame],
        in_cat:str =None,
        new_cat:Optional [str] = None, 
        **kws
        ) -> DataFrame: 
    """ Create a new category attribute to discretize instances. 
    
    A new category in data is better use to stratified the trainset and 
    the dataset to be consistent and rounding using ceil values.
    
    :param in_cat: column name used for stratified dataset 
    :param new_cat: new category name created and inset into the 
                dataframe.
    :return: new dataframe with new column of created category.
    """
    divby = kws.pop('divby', 1.5) # normalize to hold raisonable number 
    combined_cat_into = kws.pop('higherclass', 5) # upper class bound 
    
    data[new_cat]= np.ceil(data[in_cat]) /divby 
    data[new_cat].where(data[in_cat] < combined_cat_into, 
                             float(combined_cat_into), inplace =True )
    return data 

def stratify_categories(
    data: Union [ArrayLike, DataFrame],
    cat_name:str , 
    n_splits:int =1, 
    test_size:float= 0.2, 
    random_state:int = 42
    )-> Tuple[ _Sub[DataFrame[DType[_T]]], _Sub[DataFrame[DType[_T]]]]: 
    """ Stratified sampling based on new generated category. 

    :param data: dataframe holding the new column of category 
    :param cat_name: new category name inserted into `data` 
    :param n_splits: number of splits 
    """
    split = StratifiedShuffleSplit(n_splits, test_size = test_size, 
                                   random_state=random_state)
    for train_index, test_index in split.split(data, data[cat_name]): 
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index] 
        
    return strat_train_set , strat_test_set 

def fetch_model(
        modelfile: str ,
        modelpath:Optional[str] = None,
        default:bool =True,
        modname: Optional[str] =None,
        verbose:int =0): 
    """ Fetch your model saved using Python pickle module or 
    joblib module. 
    
    :param modelfile: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    :param modelpath: path-Like object , 
        Path to model dumped file =`modelfile`
    :default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters)
        . If False return all values saved from `~.MultipleGridSearch`
       
    :modname: str 
        Is the name of model to retrived from dumped file. If name is given 
        get only the model and its best parameters. 
    :verbose: int, level=0 
        control the verbosity.More message if greater than 0.
    
    :returns:
        - `model_class_params`: if default is ``True``
        - `pickedfname`: model dumped and all parameters if default is `False`
        
    :Example: 
        >>> from gofast.bases import fetch_model 
        >>> my_model = fetch_model ('SVC__LinearSVC__LogisticRegression.pkl',
                                    default =False,  modname='SVC')
        >>> my_model
    """
    
    try:
        isdir =os.path.isdir( modelpath)
    except TypeError: 
        #stat: path should be string, bytes, os.PathLike or integer, not NoneType
        isdir =False
        
    if isdir and modelfile is not None: 
        modelfile = os.join.path(modelpath, modelfile)

    isfile = os.path.isfile(modelfile)
    if not isfile: 
        raise FileNotFoundError (f"File {modelfile!r} not found!")
        
    from_joblib =False 
    if modelfile.endswith('.pkl'): from_joblib  =True 
    
    if from_joblib:
       if verbose: _logger.info(
               f"Loading models `{os.path.basename(modelfile)}`")
       try : 
           pickedfname = joblib.load(modelfile)
           # and later ....
           # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
           dmsg=f"Model {modelfile !r} retreived from~.externals.joblib`"
       except : 
           dmsg=''.join([f"Nothing to retrived. It's seems model {modelfile !r}", 
                         " not really saved using ~external.joblib module! ", 
                         "Please check your model filename."])
    
    if not from_joblib: 
        if verbose: _logger.info(
                f"Loading models `{os.path.basename(modelfile)}`")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickedfname= pickle.load (modf)
           if verbose: _logger.info(
                   f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f'Model `{modelfile!r} deserizaled from  {modelfile}`!'
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            if verbose: _logger.info(dmsg)   

    if verbose > 0: 
        pprint(
            dmsg 
            )
           
    if modname is not None: 
        keymess = f"{modname!r} not found."
        try : 
            if default:
                model_class_params  =( pickedfname[modname]['best_model'], 
                                   pickedfname[modname]['best_params_'], 
                                   pickedfname[modname]['best_scores'],
                                   )
            if not default: 
                model_class_params=pickedfname[modname]
                
        except KeyError as key_error: 
            warnings.warn(
                f"Model name {modname!r} not found in the list of dumped"
                f" models = {list(pickedfname.keys()) !r}")
            raise KeyError from key_error(keymess + "Shoud try the model's"
                                          f"names ={list(pickedfname.keys())!r}")
        
        if verbose: 
            pprint('Should return a tuple of `best model` and the'
                   ' `model best parameters.')
           
        return model_class_params  
            
    if default:
        model_class_params =list()    
        
        for mm in pickedfname.keys(): 
            model_class_params.append((pickedfname[mm]['best_model'], 
                                      pickedfname[mm]['best_params_'],
                                      pickedfname[modname]['best_scores']))
    
        if verbose: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params

    return pickedfname 

def serialize_data (
        data , 
        filename=None, 
        savepath =None, 
        to=None, 
        verbose=0,
        ): 
    """ Dump and save binary file 
    
    :param data: Object
        Object to dump into a binary file. 
    :param filename: str
        Name of file to serialize. If 'None', should create automatically. 
    :param savepath: str, PathLike object
         Directory to save file. If not exists should automaticallycreate.
    :param to: str 
        Force your data to be written with specific module like ``joblib`` or 
        Python ``pickle` module. Should be ``joblib`` or ``pypickle``.
    :return: str
        dumped or serialized filename.
        
    :Example:
        
        >>> import numpy as np
        >>> from gofast.tools.mlutils import dumpOrSerializeData
        >>>  data=(np.array([0, 1, 3]),np.array([0.2, 4]))
        >>> dumpOrSerializeData(data, filename ='__XTyT.pkl', to='pickle', 
                                savepath='gofast/datasets')
    """
    if filename is None: 
        filename ='__mydumpedfile.{}__'.format(datetime.datetime.now())
        filename =filename.replace(' ', '_').replace(':', '-')

    if to is not None: 
        if not isinstance(to, str): 
            raise TypeError(f"Need to be string format not {type(to)}")
        if to.lower().find('joblib')>=0: to ='joblib'
        elif to.lower().find('pickle')>=0:to = 'pypickle'
        
        if to not in ('joblib', 'pypickle'): 
            raise ValueError("Unknown argument `to={to}`."
                             " Should be <joblib> or <pypickle>")
    # remove extension if exists
    if filename.endswith('.pkl'): 
        filename = filename.replace('.pkl', '')
        
    if verbose: _logger.info(f'Dumping data to `{filename}`!')    
    try : 
        if to is None or to =='joblib':
            joblib.dump(data, f'{filename}.pkl')
            
            filename +='.pkl'
            _logger.info(f'Data dumped in `{filename} using '
                          'to `~.externals.joblib`!')
        elif to =='pypickle': 
            # force to move pickling data  to exception and write using 
            # Python pickle module
            raise 
    except : 
        # Now try to pickle data Serializing data 
        # Using HIGHEST_PROTOCOL is almost 2X faster and creates a file that
        # is ~10% smaller.  Load times go down by a factor of about 3X.
        with open(filename, 'wb') as wfile: 
            pickle.dump( data, wfile, protocol=pickle.HIGHEST_PROTOCOL) 
        if verbose: _logger.info( 'Data are well serialized ')
        
    if savepath is not None:
        try : 
            savepath = savepath_ (savepath)
        except : 
            savepath = savepath_ ('_dumpedData_')
        try:
            shutil.move(filename, savepath)
        except :
            print(f"--> It seems destination path {filename!r} already exists.")

    if savepath is None:
        savepath =os.getcwd()
        
    if verbose: 
        print(f"Data {'serialization' if to=='pypickle' else 'dumping'}"
          f" complete,  save to {savepath!r}")
   
def load_dumped_data (filename:str, verbose=0): 
    """ Load dumped or serialized data from filename 
    
    :param filename: str or path-like object 
        Name of dumped data file.
    :return: 
        Data loaded from dumped file.
        
    :Example:
        
        >>> from gofast.tools.mlutils import loadDumpedOrSerializedData
        >>> loadDumpedOrSerializedData(filename ='Watex/datasets/__XTyT.pkl')
    """
    
    if not isinstance(filename, str): 
        raise TypeError(f'filename should be a <str> not <{type(filename)}>')
        
    if not os.path.isfile(filename): 
        raise FileExistsError(f"File {filename!r} does not exist.")

    _filename = os.path.basename(filename)
    if verbose: _logger.info(f"Loading data from `{_filename}`!")
   
    data =None 
    try : 
        data= joblib.load(filename)
        if verbose: _logger.info(
                ''.join([f"Data from {_filename !r} are sucessfully", 
                      " loaded using ~.externals.joblib`!"]))
    except : 
        if verbose: 
            _logger.info(
            ''.join([f"Nothing to reload. It's seems data from {_filename!r}", 
                      " are not really dumped using ~external.joblib module!"])
            )
        # Try DeSerializing using pickle module
        with open(filename, 'rb') as tod: 
            data= pickle.load (tod)
            
        if verbose: 
            _logger.info(f"Data from `{_filename!r}` are well"
                      " deserialized using Python pickle module!")
        
    is_none = data is None
    if is_none: 
        print("Unable to deserialize data. Please check your file.")

    return data 

def subprocess_module_installation (module, upgrade =True ): 
    """ Install  module using subprocess.
    :param module: str, module name 
    :param upgrade:bool, install the lastest version.
    """
    import sys 
    import subprocess 
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    MOD_IMP=False 
    print(f'---> Module {module!r} installation will take a while,'
          ' please be patient...')
    cmd = f'<pip install {module}> | <python -m pip install {module}>'
    try: 

        upgrade ='--upgrade' if upgrade else ''
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        f'{module}', f'{upgrade}'])
        reqs = subprocess.check_output([sys.executable,'-m', 'pip',
                                        'freeze'])
        [r.decode().split('==')[0] for r in reqs.split()]
        _logger.info(f"Intallation of `{module}` and dependancies"
                     "was successfully done!") 
        MOD_IMP=True
     
    except: 
        _logger.error(f"Fail to install the module =`{module}`.")
        print(f'---> Module {module!r} installation failed, Please use'
           f'  the following command {cmd} to manually install it.')
    return MOD_IMP 
        
                
def _assert_sl_target (target,  df=None, obj=None): 
    """ Check whether the target name into the dataframe for supervised 
    learning.
    
    :param df: dataframe pandas
    :param target: str or index of the supervised learning target name. 
    
    :Example: 
        
        >>> from gofast.tools.mlutils import _assert_sl_target
        >>> from gofast.datasets import fetch_data
        >>> data = fetch_data('Bagoue original').get('data=df')  
        >>> _assert_sl_target (target =12, obj=prepareObj, df=data)
        ... 'flow'
    """
    is_dataframe = isinstance(df, pd.DataFrame)
    is_ndarray = isinstance(df, np.ndarray)
    if is_dataframe :
        targets = smart_format(
            df.columns if df.columns is not None else [''])
    else:targets =''
    
    if target is None:
        nameObj=f'{obj.__class__.__name__}'if obj is not None else 'Base class'
        msg =''.join([
            f"{nameObj!r} {'basically' if obj is not None else ''}"
            " works with surpervised learning algorithms so the",
            " input target is needed. Please specify the target", 
            f" {'name' if is_dataframe else 'index' if is_ndarray else ''}", 
            " to take advantage of the full functionalities."
            ])
        if is_dataframe:
            msg += f" Select the target among {targets}."
        elif is_ndarray : 
            msg += f" Max columns size is {df.shape[1]}"

        warnings.warn(msg, UserWarning)
        _logger.warning(msg)
        
    if target is not None: 
        if is_dataframe: 
            if isinstance(target, str):
                if not target in df.columns: 
                    msg =''.join([
                        f"Wrong target value {target!r}. Please select "
                        f"the right column name: {targets}"])
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg)
                    target =None
            elif isinstance(target, (float, int)): 
                is_ndarray =True 
  
        if is_ndarray : 
            _len = len(df.columns) if is_dataframe else df.shape[1] 
            m_=f"{'less than' if target >= _len  else 'greater than'}" 
            if not isinstance(target, (float,int)): 
                msg =''.join([f"Wrong target value `{target}`!"
                              f" Object type is {type(df)!r}. Target columns", 
                              " index should be given instead."])
                warnings.warn(msg, category= UserWarning)
                _logger.warning(msg)
                target=None
            elif isinstance(target, (float,int)): 
                target = int(target)
                if not 0 <= target < _len: 
                    msg =f" Wrong target index. Should be {m_} {str(_len-1)!r}."
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg) 
                    target =None
                    
            if df is None: 
                wmsg = ''.join([
                    f"No data found! `{target}` does not fit any data set.", 
                      "Could not fetch the target name.`df` argument is None.", 
                      " Need at least the data `numpy.ndarray|pandas.dataFrame`",
                      ])
                warnings.warn(wmsg, UserWarning)
                _logger.warning(wmsg)
                target =None
                
            target = list(df.columns)[target] if is_dataframe else target
            
    return target

def export_target(
    ar, /, 
    tname, 
    drop=True , 
    columns =None,
    as_frame=False 
    ): 
    """ Extract target from multidimensional array or dataframe.  
    
    Parameters 
    ------------
    ar: arraylike2d or pd.DataFrame 
      Array that supposed to contain the target value. 
      
    tname: int/str, list of int/str 
       index or the name of the target; if ``int`` is passed it should range 
       ranged less than the columns number of the array i.e. a shape[1] in 
       the case of np.ndarray. If the list of indexes or names are given, 
       the return target should be in two dimensional array. 
       
    drop: bool, default=True 
       Remove the target array in the 2D array or dataframe in the case 
       the target exists and returns a data exluding the target array. 
       
    columns: list, default=False. 
       composes the dataframe when the array is given rather than a dataframe. 
       The list of column names must match the number of columns in the 
       two dimensional array, otherwise an error occurs. 
       
    as_frame: bool, default=False, 
       returns dataframe/series or the target rather than array when the array 
       is supplied. This seems useful when column names are supplied. 
       
    Returns
    --------
    t, ar : array-like/pd.Series , array-like/pd.DataFrame 
      Return the targets and the array/dataframe of the target. 
      
    Examples 
    ---------
    >>>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.mtutils import get_target 
    >>> ar = np.random.randn ( 3,  3 )
    >>> df0 = pd.DataFrame ( ar, columns = ['x1', 'x2', 'tname'])
    >>> df= df0.copy() 
    >>> get_target (df, 'tname', drop= False )
    (      tname
     0 -0.542861
     1  0.781198,
              x1        x2     tname
     0 -1.424061 -0.493320 -0.542861
     1  0.416050 -1.156182  0.781198)
    >>> get_target (df, [ 'tname', 'x1']) # drop is True by default
    (      tname        x1
     0 -0.542861 -1.424061
     1  0.781198  0.416050,
              x2
     0 -0.493320
     1 -1.156182)
    >>> df = df0.copy() 
    >>> # when array is passed 
    >>> get_target (df.values , '2', drop= False )
    (array([[-0.54286148],
            [ 0.7811981 ]]),
     array([[-1.42406091, -0.49331988, -0.54286148],
            [ 0.41605005, -1.15618243,  0.7811981 ]]))
    >>> get_target (df.values , 'tname') # raise error 
    ValueError: 'tname' ['tname'] is not valid...
    
    """
    emsg =("Array is passed.'tname' must be a list of indexes or column names"
           " that fit the shape[axis=1] of the given array. Expect {}, got {}.")
    emsgc =("'tname' {} {} not valid. Array is passed while columns are not "
            "supplied. Expect 'tname' in the range of numbers betwen 0- {}")
    is_arr=False 
    tname =[ str(i) for i in is_iterable(
        tname, exclude_string =True, transform =True)] 
    
    if isinstance (ar, np.ndarray): 
        columns = columns or [str(i) for i in range(ar.shape[1])]
        if len(columns) < ar.shape [1]: 
            raise ValueError(emsg.format(ar.shape[1], len(tname)))
        ar = pd.DataFrame (ar, columns = columns) 
        if not existfeatures(ar, tname, error='ignore'): 
            raise ValueError(emsgc.format(tname, "is" if len(tname)==1 else "are", 
                                         len(columns)-1)
                             )
        is_arr=True if not as_frame else False 
        
    t, ar =get_target(ar, tname , inplace = drop ) 

    return (t.values, ar.values ) if is_arr  else (t, ar) 
        
def naive_data_split(
    X, y=None, *,  
    test_size =0.2, 
    target =None,
    random_state=42, 
    fetch_target =False,
    **skws): 
    """ Splitting data function naively. 
    
    Split data into the training set and test set. If target `y` is not
    given and you want to consider a specific array as a target for 
    supervised learning, just turn `fetch_target` argument to ``True`` and 
    set the `target` argument as a numpy columns index or pandas dataframe
    colums name. 
    
    :param X: np.ndarray or pd.DataFrame 
    :param y: array_like 
    :param test_size: If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split. 
    :param random_state: int, Controls the shuffling applied to the data
        before applying the split. Pass an int for reproducible output across
        multiple function calls
    :param fetch_target: bool, use to retrieve the targetted value from 
        the whole data `X`. 
    :param target: int, str 
        If int itshould be the index of the targetted value otherwise should 
        be the columns name of pandas DataFrame.
    :param skws: additional scikit-lean keywords arguments 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    :returns: list, length -List containing train-test split of inputs.
        
    :Example: 
        
        >>> from gofast.datasets import fetch_data 
        >>> data = fetch_data ('Bagoue original').get('data=df')
        >>> X, XT, y, yT= default_data_splitting(data.values,
                                     fetch_target=True,
                                     target =12 )
        >>> X, XT, y, yT= default_data_splitting(data,
                             fetch_target=True,
                             target ='flow' )
        >>> X0= data.copy()
        >>> X0.drop('flow', axis =1, inplace=True)
        >>> y0 = data ['flow']
        >>> X, XT, y, yT= default_data_splitting(X0, y0)
    """

    if fetch_target: 
        target = _assert_sl_target (target, df =X)
        s='could not be ' if target is None else 'was succesffully '
        wmsg = ''.join([
            f"Target {'index' if isinstance(target, int) else 'value'} "
            f"{str(target)!r} {s} used to fetch the `y` value from "
            "the whole data set."])
        if isinstance(target, str): 
            y = X[target]
            X= X.copy()
            X.drop(target, axis =1, inplace=True)
        if isinstance(target, (float, int)): 
            y=X[:, target]
            X = np.delete (X, target, axis =1)
        warnings.warn(wmsg, category =UserWarning)
        
    V= train_test_split(X, y, random_state=random_state, **skws) \
        if y is not None else train_test_split(
                X,random_state=random_state, **skws)
    if y is None: 
        X, XT , yT = *V,  None 
    else: 
        X, XT, y, yT= V
    
    return  X, XT, y, yT

def soft_data_split(
    X, y=None, *,
    test_size=0.2,
    target_column=None,
    random_state=42,
    extract_target=False,
    **split_kwargs
):
    """
    Splits data into training and test sets, optionally extracting a 
    target column.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data to split. If `extract_target` is True, a target column can be
        extracted from `X`.
    y : array-like, optional
        Target variable array. If None and `extract_target` is False, `X` is
        split without a target variable.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Should be
        between 0.0 and 1.0. Default is 0.2.
    target_column : int or str, optional
        Index or column name of the target variable in `X`. Used only if
        `extract_target` is True.
    random_state : int, optional
        Controls the shuffling for reproducible output. Default is 42.
    extract_target : bool, optional
        If True, extracts the target variable from `X`. Default is False.
    split_kwargs : dict, optional
        Additional keyword arguments to pass to `train_test_split`.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data arrays.

    Raises
    ------
    ValueError
        If `target_column` is not found in `X` when `extract_target` is True.

    Example
    -------
    >>> from gofast.datasets import fetch_data
    >>> data = fetch_data('Bagoue original')['data']
    >>> X, XT, y, yT = split_data(data, extract_target=True, target_column='flow')
    """

    if extract_target:
        if isinstance(X, pd.DataFrame) and target_column in X.columns:
            y = X[target_column]
            X = X.drop(columns=target_column)
        elif hasattr(X, '__array__') and isinstance(target_column, int):
            y = X[:, target_column]
            X = np.delete(X, target_column, axis=1)
        else:
            raise ValueError(f"Target column {target_column!r} not found in X.")

    if y is not None:
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, **split_kwargs)
    else:
        X_train, X_test = train_test_split(X, test_size=test_size,
                                           random_state=random_state, 
                                           **split_kwargs)
        return X_train, X_test, None, None

def load_saved_model(
    file_path: str,
    *,
    retrieve_default: bool = True,
    model_name: Optional[str] = None,
    storage_format: Optional[str] = None,
) -> Union[object, Tuple[object, Dict[str, Any]]]:
    """
    Load a saved model or data using Python's pickle or joblib module.

    Parameters
    ----------
    file_path : str
        The path to the saved model file. Supported formats are `.pkl` and `.joblib`.
    retrieve_default : bool, default=True
        If True, returns the model along with its best parameters. If False,
        returns the entire contents of the saved file.
    model_name : str, optional
        The name of the specific model to retrieve from the saved file. If None,
        the entire file content is returned.
    storage_format : str, optional
        The format used for saving the file. If None, the format is inferred
        from the file extension. Supported formats are 'joblib' and 'pickle'.

    Returns
    -------
    object or Tuple[object, Dict[str, Any]]
        The loaded model or a tuple of the model and its parameters, depending
        on the `retrieve_default` value.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If the specified model name is not found in the file.
    ValueError
        If the storage format is not supported.

    Example
    -------
    >>> from gofast.tools.mlutils import load_saved_model
    >>> model, params = load_saved_model('path_to_file.pkl', model_name='SVC')
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path!r} not found.")

    # Infer storage format from file extension if not specified
    storage_format = storage_format or os.path.splitext(
        file_path)[-1].lower().lstrip('.')
    if storage_format not in {"joblib", "pickle"}:
        raise ValueError(f"Unsupported storage format {storage_format!r}."
                         " Use 'joblib' or 'pickle'.")

    # Load the saved file
    if storage_format == 'joblib':
        loaded_data = joblib.load(file_path)
    elif storage_format == 'pickle':
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)

    # If a specific model name is provided, extract it
    if model_name:
        if model_name not in loaded_data:
            raise KeyError(f"Model {model_name!r} not found in the file."
                           f" Available models: {list(loaded_data.keys())}")
        if retrieve_default:
            return ( loaded_data[model_name]['best_model'],
                    loaded_data[model_name]['best_params_'])
        else:
            return loaded_data[model_name]

    return loaded_data

@deprecated ("Deprecated function. It should be removed soon in"
             " the next realease...")
def fetchModel(
    file: str,
    *, 
    default: bool = True,
    name: Optional[str] = None,
    storage=None, 
)-> object: 
    """ Fetch your data/model saved using Python pickle or joblib module. 
    
    Parameters 
    ------------
    file: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    path: path-Like object , 
        Path to model dumped file =`modelfile`
    default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters).
        If ``False`` return all values saved from `~.MultipleGridSearch`
    storage: str, default='joblib'
        kind of module use to pickling the data
    name: str 
        Is the name of model to retreived from dumped file. If name is given 
        get only the model and its best parameters. 
        
    Returns
    --------
    - `data`: Tuple (Dict, )
        data composed of models, classes and params for 'best_model', 
        'best_params_' and 'best_scores' if default is ``True``,
        and model dumped and all parameters otherwise.

    Example
    ---------
        >>> from gofast.bases import fetch_model 
        >>> my_model, = fetchModel ('SVC__LinearSVC__LogisticRegression.pkl',
                                    default =False,  modname='SVC')
        >>> my_model
    """
    
    if not os.path.isfile (file): 
        raise FileNotFoundError (f"File {file!r} not found. Please check"
                                 " your filename.")
    st = storage 
    if storage is None: 
        ex = os.path.splitext (file)[-1] 
        storage = 'joblib' if ex =='.joblib' else 'pickle'

    storage = str(storage).lower().strip() 
    
    assert storage in {"joblib", "pickle"}, (
        "Data pickling supports only the Python's built-in persistence"
        f" model'pickle' or 'joblib' as replacement of pickle: got{st!r}"
        )
    _logger.info(f"Loading models {os.path.basename(file)}")
    
    if storage =='joblib':
        pickledmodel = joblib.load(file)
        if len(pickledmodel)>=2 : 
            pickledmodel = pickledmodel[0]
    elif storage =='pickle': 
        with open(file, 'rb') as modf: 
            pickledmodel= pickle.load (modf)
            
    data= copy.deepcopy(pickledmodel)
    if name is not None: 
        name =_assert_all_types(name, str, objname="Model to pickle ")
        if name not in pickledmodel.keys(): 
            raise KeyError(
                f"Model {name!r} is missing in the dumped models."
                f" Available pickled models: {list(pickledmodel.keys())}"
                         )
        if default: 
            data =[pickledmodel[name][k] for k in (
                "best_model", "best_params_", "best_scores")
                ]
        else:
            # When using storage as joblib
            # trying to unpickle estimator directly other
            # format than dict from version 1.1.1 
            # might lead to breaking code or invalid results. 
            # Use at your own risk. For more info please refer to:
            # https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
            
            # pickling all data
            data= pickledmodel.get(name)
        
    return data,       

def find_features_in( 
    df: DataFrame= None, 
    features: List[str]= None,  
    parse_features: bool=False, 
    return_frames: bool= False, 
    ) -> Tuple[List[str] | DataFrame, List[str] |DataFrame]: 
    """ 
    Retrieve the categorial or numerical features on whole features 
    of dataset. 
    
    Parameters 
    -----------
    df: Dataframe 
        Dataframe with columns composing the features
        
    features: list of str, 
        list of the column names. If the dataframe is big, can set the only 
        required features. If features are provided, frame should be shrunked 
        to match the only given features before the numerical and categorical 
        features search. Note that an error will raises if any of one features 
        is missing in the dataframe. 
        
    return_frames: bool, 
        if set to ``True``, it returns two separated dataframes (cat & num) 
        otherwise, it only returns the cat and num columns names. 
    parse_features: bool, default=False, 
       Use default parsers to parse string items into an interable object. 
       
    Returns
    ---------
    Tuple:  `cat_features` and  `num_features` names or frames 
       
    Examples 
    ----------
    >>> from gofast.datasets import fetch_data 
    >>>> from gofast.tools.mlutils import find_features_in
    >>> data = fetch_data ('bagoue original').get('data=dfy2')
    >>> cat, num = find_features_in(data)
    >>> cat, num 
    ... (['type', 'geol', 'shape', 'name', 'flow'],
     ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(
        data, features = ['geol', 'ohmS', 'sfi'])
    ... (['geol'], ['ohmS', 'sfi'])
        
    """
    if not is_frame (df, df_only =True ):
        raise TypeError(
            f"Expect a dataframe. Got {type(df).__name__!r}")
    if features is not None: 
        features = list( is_iterable(
            features, exclude_string= True, transform =True, 
            parse_string= parse_features) 
                        )
    if features is None: # get  the whole features 
        features = list(df.columns) 
        
    existfeatures(df, list(features))
    df = df[features].copy() 
    
    # get num features 
    num = select_features(df, include = 'number')
    catnames = findDifferenceGenObject (df.columns, num.columns ) 
    if catnames is None: catnames =[]
    return ( df[catnames], num) if return_frames else (
        list(catnames), list(num.columns)  )
   
def categorize_target(
    arr :ArrayLike |Series , /, 
    func: _F = None,  
    labels: int | List[int] = None, 
    rename_labels: Optional[str] = None, 
    coerce:bool=False,
    order:str='strict',
    ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from gofast.tools.mlutils import categorize_target 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = categorize_target(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> categorize_target(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> categorize_target(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> categorize_target(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> categorize_target(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (
        arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`gofast.tools.mlutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to be categorized. If None or any other values, 
        the categorization of labels considers only the length of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly considers the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)       
        
def projection_validator (X, Xt=None, columns =None ):
    """ Retrieve x, y coordinates of a datraframe ( X, Xt ) from columns 
    names or indexes. 
    
    If X or Xt are given as arrays, `columns` may hold integers from 
    selecting the the coordinates 'x' and 'y'. 
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to consider as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
      
    Returns 
    -------
    ( x, y, xt, yt ), (xname, yname, xtname, ytname), Tuple of coordinate 
        arrays and coordinate labels 
 
    """
    # initialize arrays and names 
    init_none = [None for i in range (4)]
    x,y, xt, yt = init_none
    xname,yname, xtname, ytname = init_none 
    
    m="{0} must be an iterable object, not {1!r}"
    ms= ("{!r} is given while columns are not supplied. set the list of "
        " feature names or indexes to fetch 'x' and 'y' coordinate arrays." )
    
    # args = list(args) + [None for i in range (5)]
    # x, y, xt, yt, *_ = args 
    X =_assert_all_types(X, np.ndarray, pd.DataFrame ) 
    
    if Xt is not None: 
        Xt = _assert_all_types(Xt, np.ndarray, pd.DataFrame)
        
    if columns is not None: 
        if isinstance (columns, str): 
            columns = str2columns(columns )
        
        if not is_iterable(columns): 
            raise ValueError(m.format('columns', type(columns).__name__))
        
        columns = list(columns) + [ None for i in range (5)]
        xname , yname, xtname, ytname , *_= columns 

    if isinstance(X, pd.DataFrame):
      
        x, xname, y, yname = _validate_columns(X, xname, yname)
        
    elif isinstance(X, np.ndarray):
        x, y = _is_valid_coordinate_arrays (X, xname, yname )    
        
        
    if isinstance (Xt, pd.DataFrame) :
        # the test set holds the same feature names
        # as the train set 
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, xtname, yt, ytname = _validate_columns(Xt, xname, yname)

    elif isinstance(Xt, np.ndarray):
        
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, yt = _is_valid_coordinate_arrays (Xt, xtname, ytname , 'test')
        
    if (x is None) or (y is None): 
        raise ValueError (ms.format('X'))
    if Xt is not None: 
        if (xt is None) or (yt is None): 
            warnings.warn (ms.format('Xt'))

    return  (x, y , xt, yt ) , (
        xname, yname, xtname, ytname ) 
    

def _validate_columns (df, xni, yni ): 
    """ Validate the feature name  in the dataframe using either the 
    string litteral name of the index position in the columns.
    
    :param df: pandas.DataFrame- Dataframe with feature names as columns. 
    :param xni: str, int- feature name  or position index in the columns for 
        x-coordinate 
    :param yni: str, int- feature name  or position index in the columns for 
        y-coordinate 
    
    :returns: (x, ni) Tuple of (pandas.Series, and names) for x and y 
        coordinates respectively.
    
    """
    def _r (ni): 
        if isinstance(ni, str): # feature name
            existfeatures(df, ni ) 
            s = df[ni]  
        elif isinstance (ni, (int, float)):# feature index
            s= df.iloc[:, int(ni)] 
            ni = s.name 
        return s, ni 
        
    xs , ys = [None, None ]
    if df.ndim ==1: 
        raise ValueError ("Expect a dataframe of two dimensions, got '1'")
        
    elif df.shape[1]==2: 
       warnings.warn("columns are not specify while array has dimension"
                     "equals to 2. Expect indexes 0 and 1 for (x, y)"
                     "coordinates respectively.")
       xni= df.iloc[:, 0].name 
       yni= df.iloc[:, 1].name 
    else: 
        ms = ("The matrix of features is greater than 2. Need column names or"
              " indexes to  retrieve the 'x' and 'y' coordinate arrays." ) 
        e =' Only {!r} is given.' 
        me=''
        if xni is not None: 
            me =e.format(xni)
        if yni is not None: 
            me=e.format(yni)
           
        if (xni is None) or (yni is None ): 
            raise ValueError (ms + me)
            
    xs, xni = _r (xni) ;  ys, yni = _r (yni)
  
    return xs, xni , ys, yni 


def _validate_array_indexer (arr, index): 
    """ Select the appropriate coordinates (x,y) arrays from indexes.  
    
    Index is used  to retrieve the array of (x, y) coordinates if dimension 
    of `arr` is greater than 2. Since we expect x, y coordinate for projecting 
    coordinates, 1-d  array `X` is not acceptable. 
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
    :param index: int, index to fetch x, and y coordinates in multi-dimension
        arrays. 
    :returns: arr- x or y coordinates arrays. 

    """
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
    if not isinstance (index, (float, int)): 
        raise ValueError("index is needed to coordinate array with "
                         "dimension greater than 2.")
        
    return arr[:, int (index) ]

def _is_valid_coordinate_arrays (arr, xind, yind, ptype ='train'): 
    """ Check whether array is suitable for projecting i.e. whether 
    x and y (both coordinates) can be retrived from `arr`.
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
        
    :param xind: int, index to fetch x-coordinate in multi-dimension
        arrays. 
    :param yind: int, index to fetch y-coordinate in multi-dimension
        arrays
    :param ptype: str, default='train', specify whether the array passed is 
        training or test sets. 
    :returns: (x, y)- array-like of x and y coordinates. 
    
    """
    xn, yn =('x', 'y') if ptype =='train' else ('xt', 'yt') 
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
        
    elif arr.shape[1] ==2 : 
        x, y = arr[:, 0], arr[:, 1]
        
    else :
        msg=("The matrix of features is greater than 2; Need index to  "
             " retrieve the {!r} coordinate array in param 'column'.")
        
        if xind is None: 
            raise ValueError(msg.format(xn))
        else : x = _validate_array_indexer(arr, xind)
        if yind is None : 
            raise ValueError(msg.format(yn))
        else : y = _validate_array_indexer(arr, yind)
        
    return x, y         
        
def labels_validator (t, /, labels, return_bool = False): 
    """ Assert the validity of the label in the target  and return the label 
    or the boolean whether all items of label are in the target. 
    
    :param t: array-like, target that is expected to contain the labels. 
    :param labels: int, str or list of (str or int) that is supposed to be in 
        the target `t`. 
    :param return_bool: bool, default=False; returns 'True' or 'False' rather 
        the labels if set to ``True``. 
    :returns: bool or labels; 'True' or 'False' if `return_bool` is set to 
        ``True`` and labels otherwise. 
        
    :example: 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.tools.mlutils import cattarget, labels_validator 
    >>> _, y = fetch_data ('bagoue', return_X_y=True, as_frame=True) 
    >>> # binarize target y into [0 , 1]
    >>> ybin = cattarget(y, labels=2 )
    >>> labels_validator (ybin, [0, 1])
    ... [0, 1] # all labels exist. 
    >>> labels_validator (y, [0, 1, 3])
    ... ValueError: Value '3' is missing in the target.
    >>> labels_validator (ybin, 0 )
    ... [0]
    >>> labels_validator (ybin, [0, 5], return_bool=True ) # no raise error
    ... False
        
    """
    
    if not is_iterable(labels):
        labels =[labels] 
        
    t = np.array(t)
    mask = _isin(t, labels, return_mask=True ) 
    true_labels = np.unique (t[mask]) 
    # set the difference to know 
    # whether all labels are valid 
    remainder = list(set(labels).difference (true_labels))
    
    isvalid = True 
    if len(remainder)!=0 : 
        if not return_bool: 
            # raise error  
            raise ValueError (
                "Label value{0} {1} {2} missing in the target 'y'.".format ( 
                f"{'s' if len(remainder)>1 else ''}", 
                f"{smart_format(remainder)}",
                f"{'are' if len(remainder)> 1 else 'is'}")
                )
        isvalid= False 
        
    return isvalid if return_bool else  labels 
        
def bi_selector (d, /,  features =None, return_frames = False,
                 parse_features:bool=... ):
    """ Auto-differentiates the numerical from categorical attributes.
    
    This is usefull to select the categorial features from the numerical 
    features and vice-versa when we are a lot of features. Enter features 
    individually become tiedous and a mistake could probably happenned. 
    
    Parameters 
    ------------
    d: pandas dataframe 
        Dataframe pandas 
    features : list of str
        List of features in the dataframe columns. Raise error is feature(s) 
        does/do not exist in the frame. 
        Note that if `features` is ``None``, it returns the categorical and 
        numerical features instead. 
        
    return_frames: bool, default =False 
        return the difference columns (features) from the given features  
        as a list. If set to ``True`` returns bi-frames composed of the 
        given features and the remaining features. 
        
    Returns 
    ----------
    - Tuple ( list, list)
        list of features and remaining features 
    - Tuple ( pd.DataFrame, pd.DataFrame )
        List of features and remaing features frames.  
            
    Example 
    --------
    >>> from gofast.tools.mlutils import bi_selector 
    >>> from gofast.datasets import load_hlogs 
    >>> data = load_hlogs().frame # get the frame 
    >>> data.columns 
    >>> Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter', 'aquifer_group',
           'pumping_level', 'aquifer_thickness', 'hole_depth_before_pumping',
           'hole_depth_after_pumping', 'hole_depth_loss', 'depth_starting_pumping',
           'pumping_depth_at_the_end', 'pumping_depth', 'section_aperture', 'k',
           'kp', 'r', 'rp', 'remark'],
          dtype='object')
    >>> num_features, cat_features = bi_selector (data)
    >>> num_features
    ...['gamma_gamma',
         'depth_top',
         'aquifer_thickness',
         'pumping_depth_at_the_end',
         'section_aperture',
         'remark',
         'depth_starting_pumping',
         'hole_depth_before_pumping',
         'rp',
         'hole_depth_after_pumping',
         'hole_depth_loss',
         'depth_bottom',
         'sp',
         'pumping_depth',
         'kp',
         'resistivity',
         'short_distance_gamma',
         'r',
         'natural_gamma',
         'layer_thickness',
         'k',
         'well_diameter']
    >>> cat_features 
    ... ['hole_id', 'strata_name', 'rock_name', 'aquifer_group', 
         'pumping_level']
    """
    parse_features, = ellipsis2false(parse_features )
    _assert_all_types( d, pd.DataFrame, objname=" unfunc'bi-selector'")
    if features is None: 
        d, diff_features, features = to_numeric_dtypes(
            d,  return_feature_types= True ) 
    if features is not None: 
        features = is_iterable(features, exclude_string= True, transform =True, 
                               parse_string=parse_features )
        diff_features = is_in_if( d.columns, items =features, return_diff= True )
        if diff_features is None: diff_features =[]
    return  ( diff_features, features ) if not return_frames else  (
        d [diff_features] , d [features ] ) 

def make_pipe(
    X, 
    y =None, *,   
    num_features = None, 
    cat_features=None, 
    label_encoding='LabelEncoder', 
    scaler = 'StandardScaler' , 
    missing_values =np.nan, 
    impute_strategy = 'median', 
    sparse_output=True, 
    for_pca =False, 
    transform =False, 
    ): 
    """ make a pipeline to transform data at once. 
    
    make a naive pipeline is usefull to fast preprocess the data at once 
    for quick prediction. 
    
    Work with a pandas dataframe. If `None` features is set, the numerical 
    and categorial features are automatically retrieved. 
    
    Parameters
    ---------
    X : pandas dataframe of shape (n_samples, n_features)
        The input samples. Use ``dtype=np.float32`` for maximum
        efficiency. Sparse matrices are also supported, use sparse
        ``csc_matrix`` for maximum efficiency.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    num_features: list or str, optional 
        Numerical features put on the list. If `num_features` are given  
        whereas `cat_features` are ``None``, `cat_features` are figured out 
        automatically.
    cat_features: list of str, optional 
        Categorial features put on the list. If `num_features` are given 
        whereas `num_features` are ``None``, `num_features` are figured out 
        automatically.
    label_encoding: callable or str, default='sklearn.preprocessing.LabelEncoder'
        kind of encoding used to encode label. This assumes 'y' is supplied. 
    scaler: callable or str , default='sklearn.preprocessing.StandardScaler'
        kind of scaling used to scaled the numerical data. Note that for 
        the categorical data encoding, 'sklearn.preprocessing.OneHotEncoder' 
        is implemented  under the hood instead. 
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    
    impute_strategy : str, default='mean'
        The imputation strategy.
    
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
    
           strategy="constant" for fixed value imputation.
           
    sparse_output : bool, default=False
        Is used when label `y` is given. Binarize labels in a one-vs-all 
        fashion. If ``True``, returns array from transform is desired to 
        be in sparse CSR format.
        
    for_pca:bool, default=False, 
        Transform data for principal component ( PCA) analysis. If set to 
        ``True``, :class:`sklearn.preprocessing.OrdinalEncoder`` is used insted 
        of :class:sklearn.preprocessing.OneHotEncoder``. 
        
    transform: bool, default=False, 
        Tranform data inplace rather than returning the naive pipeline. 
        
    Returns
    ---------
    full_pipeline: :class:`gofast.exlib.sklearn.FeatureUnion`
        - Full pipeline composed of numerical and categorical pipes 
    (X_transformed &| y_transformed):  {array-like, sparse matrix} of \
        shape (n_samples, n_features)
        - Transformed data. 
        
        
    Examples 
    ---------
    >>> from gofast.tools.mlutils import make_naive_pipe 
    >>> from gofast.datasets import load_hlogs 
    
    (1) Make a naive simple pipeline  with RobustScaler, StandardScaler 
    >>> from gofast.exlib.sklearn import RobustScaler 
    >>> X_, y_ = load_hlogs (as_frame=True )# get all the data  
    >>> pipe = make_naive_pipe(X_, scaler =RobustScaler ) 
    
    (2) Transform X in place with numerical and categorical features with 
    StandardScaler (default). Returned CSR matrix 
    
    >>> make_naive_pipe(X_, transform =True )
    ... <181x40 sparse matrix of type '<class 'numpy.float64'>'
    	with 2172 stored elements in Compressed Sparse Row format>

    """
    
    from ..transformers import DataFrameSelector
    
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not hasattr (X, '__array__'):
        raise TypeError(f"'make_naive_pipe' not supported {type(X).__name__!r}."
                        " Expects X as 'pandas.core.frame.DataFrame' object.")
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="Array for transforming X or making naive pipeline"
        )
    if not hasattr (X, "columns"):
        # create naive column for 
        # Dataframe selector 
        X = pd.DataFrame (
            X, columns = [f"naive_{i}" for i in range (X.shape[1])]
            )
    #-> Encode y if given
    if y is not None: 
        # if (label_encoding =='labelEncoder'  
        #     or get_estimator_name(label_encoding) =='LabelEncoder'
        #     ): 
        #     enc =LabelEncoder()
        if  ( label_encoding =='LabelBinarizer' 
                or get_estimator_name(label_encoding)=='LabelBinarizer'
               ): 
            enc =LabelBinarizer(sparse_output=sparse_output)
        else: 
            label_encoding =='labelEncoder'
            enc =LabelEncoder()
            
        y= enc.fit_transform(y)
    #set features
    if num_features is not None: 
        cat_features, num_features  = bi_selector(
            X, features= num_features 
            ) 
    elif cat_features is not None: 
        num_features, cat_features  = bi_selector(
            X, features= cat_features 
            )  
    if ( cat_features is None 
        and num_features is None 
        ): 
        num_features , cat_features = bi_selector(X ) 
    # assert scaler value 
    if get_estimator_name (scaler)  in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler )) 
    elif ( any ( [v.lower().find (str(scaler).lower()) >=0
                  for v in sc.keys()])
          ):  
        for k, v in sc.items () :
            if k.lower().find ( str(scaler).lower() ) >=0: 
                scaler = v ; break 
    else : 
        msg = ( f"Supports {smart_format( sc.keys(), 'or')} or "
                "other scikit-learn scaling objects, got {!r}" 
                )
        if hasattr (scaler, '__module__'): 
            name = getattr (scaler, '__module__')
            if getattr (scaler, '__module__') !='sklearn.preprocessing._data':
                raise ValueError (msg.format(name ))
        else: 
            name = scaler.__name__ if callable (scaler) else (
                scaler.__class__.__name__ ) 
            raise ValueError (msg.format(name ))
    # make pipe 
    npipe = [
            ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=impute_strategy)),                
            ('scalerObj', scaler() if callable (scaler) else scaler ), 
            ]
    
    if len(num_features)!=0 : 
       npipe.insert (
            0,  ('selectorObj', DataFrameSelector(columns= num_features))
            )

    num_pipe=Pipeline(npipe)
    
    if for_pca : 
        encoding=  ('OrdinalEncoder',OrdinalEncoder())
    else:  encoding =  (
        'OneHotEncoder', OneHotEncoder())
        
    cpipe = [
        encoding
        ]
    if len(cat_features)!=0: 
        cpipe.insert (
            0, ('selectorObj', DataFrameSelector(columns= cat_features))
            )

    cat_pipe = Pipeline(cpipe)
    # make transformer_list 
    transformer_list = [
        ('num_pipeline', num_pipe),
        ('cat_pipeline', cat_pipe), 
        ]

    #remove num of cat pipe if one of them is 
    # missing in the data 
    if len(cat_features)==0: 
        transformer_list.pop(1) 
    if len(num_features )==0: 
        transformer_list.pop(0)
        
    full_pipeline =FeatureUnion(transformer_list=transformer_list) 
    
    return  ( full_pipeline.fit_transform (X) if y is None else (
        full_pipeline.fit_transform (X), y ) 
             ) if transform else full_pipeline
       
def build_data_preprocessor(
    X, y=None, *,  
    num_features=None, 
    cat_features=None, 
    custom_transformers=None,
    label_encoding='LabelEncoder', 
    scaler='StandardScaler', 
    missing_values=np.nan, 
    impute_strategy='median', 
    feature_interaction=False,
    dimension_reduction=None,
    feature_selection=None,
    balance_classes=False,
    advanced_imputation=None,
    verbose=False,
    output_format='array',
    transform=False,
    **kwargs
    ):
    """
    Create a preprocessing pipeline for data transformation and feature engineering.

    This function constructs a pipeline to preprocess data for machine learning tasks, 
    accommodating a variety of transformations including scaling, encoding, 
    and dimensionality reduction. It supports both numerical and categorical data, 
    and can incorporate custom transformations.

    Parameters
    ----------
    X : DataFrame
        Input features dataframe.
    y : array-like, optional
        Target variable. Required for supervised learning tasks.
    num_features : list of str, optional
        List of numerical feature names. If None, determined automatically.
    cat_features : list of str, optional
        List of categorical feature names. If None, determined automatically.
    custom_transformers : list of tuples, optional
        Custom transformers to be included in the pipeline. Each tuple should 
        contain ('name', transformer_instance).
    label_encoding : str or transformer, default 'LabelEncoder'
        Encoder for the target variable. Accepts standard scikit-learn encoders 
        or custom encoder objects.
    scaler : str or transformer, default 'StandardScaler'
        Scaler for numerical features. Accepts standard scikit-learn scalers 
        or custom scaler objects.
    missing_values : int, float, str, np.nan, None, default np.nan
        Placeholder for missing values for imputation.
    impute_strategy : str, default 'median'
        Imputation strategy. Options: 'mean', 'median', 'most_frequent', 'constant'.
    feature_interaction : bool, default False
        If True, generate polynomial and interaction features.
    dimension_reduction : str or transformer, optional
        Dimensionality reduction technique. Accepts 'PCA', 't-SNE', or custom object.
    feature_selection : str or transformer, optional
        Feature selection method. Accepts 'SelectKBest', 'SelectFromModel', or custom object.
    balance_classes : bool, default False
        If True, balance classes in classification tasks.
    advanced_imputation : transformer, optional
        Advanced imputation technique like KNNImputer or IterativeImputer.
    verbose : bool, default False
        Enable verbose output.
    output_format : str, default 'array'
        Desired output format: 'array' or 'dataframe'.
    transform : bool, default False
        If True, apply the pipeline to the data immediately and return transformed data.

    Returns
    -------
    full_pipeline : Pipeline or (X_transformed, y_transformed)
        The constructed preprocessing pipeline, or transformed data if `transform` is True.

    Examples
    --------
    >>> from gofast.tools.mlutils import build_data_preprocessor
    >>> from gofast.datasets import load_hlogs
    >>> X, y = load_hlogs(as_frame=True)
    >>> pipeline = build_data_preprocessor(X, y, scaler='RobustScaler')
    >>> X_transformed = pipeline.fit_transform(X)
    
    """
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}
    # assert scaler value 
    if get_estimator_name (scaler) in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler ))() 
        
    # Define numerical and categorical pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy, missing_values=missing_values)),
        ('scaler', StandardScaler() if scaler == 'StandardScaler' else scaler)
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', missing_values=missing_values)),
        ('encoder', OneHotEncoder() if label_encoding == 'LabelEncoder' else label_encoding)
    ])

    # Determine automatic feature selection if not provided
    if num_features is None and cat_features is None:
        num_features = make_column_selector(dtype_include=['int', 'float'])(X)
        cat_features = make_column_selector(dtype_include='object')(X)

    # Feature Union for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_features),
            ('cat', categorical_pipeline, cat_features)
        ])

    # Add custom transformers if any
    if custom_transformers:
        for name, transformer in custom_transformers:
            preprocessor.named_transformers_[name] = transformer

    # Feature interaction, selection, and dimension reduction
    steps = [('preprocessor', preprocessor)]
    
    if feature_interaction:
        steps.append(('interaction', PolynomialFeatures()))
    if feature_selection:
        steps.append(('feature_selection', SelectKBest() 
                      if feature_selection == 'SelectKBest' else feature_selection))
    if dimension_reduction:
        steps.append(('dim_reduction', PCA() if dimension_reduction == 'PCA' 
                      else dimension_reduction))

    # Final pipeline
    pipeline = Pipeline(steps)

   # Advanced imputation logic if required
    if advanced_imputation:
        if advanced_imputation == 'IterativeImputer':
            steps.insert(0, ('advanced_imputer', IterativeImputer(
                estimator=RandomForestClassifier(), random_state=42)))
        else:
            steps.insert(0, ('advanced_imputer', advanced_imputation))

    # Final pipeline before class balancing
    pipeline = Pipeline(steps)

    # Class balancing logic if required
    if balance_classes and y is not None:
        if str(balance_classes).upper() == 'SMOTE':
            msg =(" Missing 'imblearn'package. 'SMOTE` can't be used. Note that"
                  " `imblearn` is the shorthand of the package 'imbalanced-learn'."
                  " Use `pip install imbalanced-learn` instead.")
            import_optional_dependency("imblearn", extra = msg )
            #-----------------------------------------
            from imblearn.over_sampling import SMOTE
            #-----------------------------------------
            # Note: SMOTE works on numerical data, so it's applied after initial pipeline
            pipeline = Pipeline([('preprocessing', pipeline), (
                'smote', SMOTE(random_state=42))])

    # Transform data if transform flag is set
    if transform:
        if output_format == 'dataframe':
            return pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)
        else:
            return pipeline.fit_transform(X)

    return pipeline
 
def select_feature_importances (
    clf, 
    X, 
    y=None, *,  
    threshold = .1 , 
    prefit = True , 
    verbose = 0 ,
    return_selector =False, 
    **kws
    ): 
    """
    Select feature importance  based on a user-specified threshold 
    after model fitting. 
    
    This is useful if one want to use `RandomForestClassifier` as a feature 
    selector and intermediate step in scikit-learn ``Pipeline`` object, which 
    allows us to connect different processing steps  with an estimator. 
  
    Parameters 
    ----------
    clf : estimator object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y: array-like of shape (n_samples, ) 
        Target vector where `n_samples` is the number of samples. If given, 
        set `prefit=False` for estimator to fit and transform the data for 
        feature importance selecting. If estimator is already fitted  i.e.
        `prefit=True`, 'y' is not needed.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        absolute importance value is greater or equal are kept while the others
        are discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted and updated by calling
        `fit` and `partial_fit`, respectively.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.
    
    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed by using the output of `max_feaures(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.
        
    return_selector: bool, default=False, 
        Returns selector object if ``True``., otherwise returns the transformed
        `X`. 
        
    verbose: int, default=0 
        display the number of features that meet the criterion according to 
        their importance range. 
    
    Returns 
    --------
    Xs or selector : ndarray (n_samples, n_criterion_features), or \
        :class:`sklearn.feature_selection.SelectFromModel`
        Ndarray of number of samples and features that meet the criterion
        according to the importance range or selector object 
        
        
    Examples
    --------
    >>> from gofast.tools.mlutils import select_feature_importances
    >>> from gofast.exlib.sklearn import LogisticRegression
    >>> X0 = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y0 = [0, 1, 0, 1]
    
    (1) use prefit =True and get the Xs importance features 
    >>> Xs = select_feature_importances (
        LogisticRegression().fit(X0, y0), 
        X0 , prefit =True )
    >>> Xs 
    array([[ 0.87, -1.34,  0.31],
           [-2.79, -0.02, -0.85],
           [-1.34, -0.48, -2.55],
           [ 1.92,  1.48,  0.65]])
    
    (2) Set off prefix  and return selector obj 
    
    >>> selector= select_feature_importances (
        LogisticRegression(), X= X0 , 
        y =y0  ,
        prefit =False , return_selector= True 
        )
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.1
    >>> selector.get_support()
    array([ True,  True,  True])
    
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.55245...
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform (X0) 
    array([[ 0.87, -1.34,  0.31],
           [-2.79, -0.02, -0.85],
           [-1.34, -0.48, -2.55],
           [ 1.92,  1.48,  0.65]])
    
    """
    if ( hasattr (clf, 'feature_names_in_') 
        or hasattr(clf, "feature_importances_")
        or hasattr (clf, 'coef_')
        ): 
        if not prefit: 
            warnings.warn(f"It seems the estimator {get_estimator_name (clf)!r}"
                          "is fitted. 'prefit' is set to 'True' to call "
                          "transform directly.")
            prefit =True 
            
    selector = SelectFromModel(
        clf, 
        threshold= threshold , 
        prefit= prefit, 
        **kws
        )
    
    if prefit:
        Xs = selector.transform(X) 
    else:
        Xs = selector.fit_transform(X, y =y)
        
    if verbose: 
        print(f"Number of features that meet the 'threshold={threshold}'" 
              " criterion: ", Xs.shape[1]
              ) 
        
    return selector if return_selector else Xs 

 
def soft_imputer (
    X, 
    y=None, 
    strategy = 'mean', 
    mode=None,  
    drop_features =False,  
    missing_values= np.nan ,
    fill_value = None , 
    verbose = "deprecated",
    add_indicator = False,  
    copy = True, 
    keep_empty_features=False, 
    **fit_params 
 ): 
    """ Imput missing values in the data. 
    
    Whatever data contains categorial features, 'bi-impute' argument passed to 
    'kind' parameters has a strategy to both impute the numerical and 
    categorical features rather than raising an error when the 'strategy' is 
    not set to 'most_frequent'.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to compute the mean and standard deviation
        used for later scaling along the features axis.
        
    y : None
        Not used, present here for API consistency by convention.
        
    strategy : str, default='mean'
       The imputation strategy.

       - If "mean", then replace missing values using the mean along
         each column. Can only be used with numeric data.
       - If "median", then replace missing values using the median along
         each column. Can only be used with numeric data.
       - If "most_frequent", then replace missing using the most frequent
         value along each column. Can be used with strings or numeric data.
         If there is more than one such value, only the smallest is returned.
       - If "constant", then replace missing values with fill_value. Can be
         used with strings or numeric data.

          strategy="constant" for fixed value imputation.
        
    mode: str, [bi-impute'], default= None
        If mode is set to 'bi-impute', it imputes the both numerical and 
        categorical features and returns a single imputed 
        dataframe.
        
    drop_features: bool or list, default =False, 
        drop a list of features in the dataframe before imputation. 
        If ``True`` and no list of features is supplied, the categorial 
        features are dropped. 
        
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.
        
    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when `strategy="constant"`
        in which case `fill_value` will be used instead.

        .. versionadded:: 0.2.0
         
    verbose : int, default=0
        Controls the verbosity of the imputer.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.
        
    fit_params: dict, 
        keywords arguments passed to the scikit-learn fitting parameters 
        More details on https://scikit-learn.org/stable/ 
    Returns 
    --------
    Xi: Dataframe, array-like, sparse matrix of shape (n_samples, n_features)
        Data imputed 
        
    Examples 
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import soft_imputer 
    >>> X= np.random.randn ( 7, 4 ) 
    >>> X[3, :] =np.nan  ; X[:, 3][-4:]=np.nan 
    >>> soft_imputer  (X)
    ... array([[ 1.34783528,  0.53276798, -1.57704281,  0.43455785],
               [ 0.36843174, -0.27132106, -0.38509441, -0.29371997],
               [-1.68974996,  0.15268509, -2.54446498,  0.18939122],
               [ 0.06013775,  0.36687602, -0.21973368,  0.11007637],
               [-0.27129147,  1.18103398,  1.78985393,  0.11007637],
               [ 1.09223954,  0.12924661,  0.52473794,  0.11007637],
               [-0.48663864,  0.47684353,  0.87360825,  0.11007637]])
    >>> frame = pd.DataFrame (X, columns =['a', 'b', 'c', 'd']  ) 
    >>> # change [bc] types to categorical values.
    >>> frame['b']=['pineaple', '', 'cabbage', 'watermelon', 'onion', 
                    'cabbage', 'onion']
    >>> frame['c']=['lion', '', 'cat', 'cat', 'dog', '', 'mouse']
    >>> soft_imputer(frame, kind ='bi-impute')
    ...             b      c         a         d
        0    pineaple   lion  1.347835  0.434558
        1     cabbage    cat  0.368432 -0.293720
        2     cabbage    cat -1.689750  0.189391
        3  watermelon    cat  0.060138  0.110076
        4       onion    dog -0.271291  0.110076
        5     cabbage    cat  1.092240  0.110076
        6       onion  mouse -0.486639  0.110076
        
    """
    X_cat, _isframe =None , True  
    
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="X"
        )
 
    if drop_features :
        if not hasattr(X, 'columns'): 
            raise ValueError ("Drop feature is possible only if  X is a"
                              f" dataframe. Got {type(X).__name__!r}") 
        
        if ( str(drop_features).lower().find ('cat') >=0 
                or  str(drop_features).lower()=='true' 
                    ) :
            # drop cat features
            X= to_numeric_dtypes(X, pop_cat_features=True, verbose =True )

        else : 
            if not is_iterable(drop_features): 
                raise TypeError ("Expects a list of features to drop;"
                                 " not {type(drop_features).__name__!r}")
        # drop_feature is a list assert whether features exist in X
            existfeatures(X, features = drop_features ) 
            diff_features = is_in_if(X.columns, drop_features, return_diff= True
                                     )
            if diff_features is None:
                raise DatasetError(
                    "It seems all features in X have been dropped. "
                    "Cannot impute a dataset with no features."
                    f" Drop features: '{drop_features}'")
                
            X= X[diff_features ]
            
    # ====> implement bi-impute strategy.  
    # strategy expects at the same time 
    # categorical  and num features 
    err_msg =(". Use 'bi-impute' strategy passed to"
              " the parameter 'mode' to coerce the categorical"
              " besides the numerical features."
    )
    if strategy =="most_frequent": 
       # altered the bi-impute strategy 
       # since most_frequent imputes at 
       # the same time num and cat features 
       
       mode =None 
    if mode is not None: 
        mode = str(mode).lower().strip () 
        if ( mode.find ('bi-')>=0
            or mode.find( 'bii')>=0 
            or mode.find('bim')>=0
            ): 
            mode='bi-impute'
            
        assert mode in {'bi-impute'} , (
            f"Strategy passed to 'mode' supports only 'bi-impute', not {mode!r}")

    if mode=='bi-impute':
        if not hasattr (X, 'columns'): 
            # "In pratice, the bi-Imputation is only allowed"
            # " with adataframe so create naive columns rather"
            # than raise error
            X= pd.DataFrame(X, columns =[f"bi_{i}" for i in range(X.shape[1])]
                            )
            _isframe =False 
            
        # recompute the num and cat features
        # since drop features can remove the
        # the cat features 
        X , nf, cf = to_numeric_dtypes(X, return_feature_types= True ) 
        if (len(nf) and len(cf) ) !=0 :
            # keep strategy to bi-impute 
            mode='bi-impute'
            X_cat , X = X [cf] ,  X[nf] 
            
        elif len(nf) ==0 and len(cf)!=0: 
            strategy ='most_frequent'
            mode =None # reset the kind method 
            X = X [cf]
        else: # if numeric 
            mode =None 
            
    # <==== end bi-impute strategy
    imp = SimpleImputer(strategy= strategy , 
                        missing_values= missing_values , 
                        fill_value = fill_value , 
                        # verbose = verbose, 
                        add_indicator=False, 
                        copy = copy, 
                        keep_empty_features=keep_empty_features, 
                        )
    try : 
        Xi = imp.fit_transform (X, y =y, **fit_params )
    except Exception as err :
        #improve error msg 
        raise ValueError (str(err) + err_msg)

    if hasattr (imp , 'feature_names_in_'): 
        Xi = pd.DataFrame( Xi , columns = imp.feature_names_in_)  
    # commonly when strategy is most frequent
    # categorical features are also imputed.
    # so dont need to use bi-impute strategy
    if  mode=='bi-impute':
        imp.strategy ='most_frequent'
        Xi_cat  = imp.fit_transform (X_cat, y =y, **fit_params ) 
        Xi_cat = pd.DataFrame( Xi_cat , columns = imp.feature_names_in_)
        Xi = pd.concat ([Xi_cat, Xi], axis =1 )
        
        if not _isframe : 
            Xi = Xi.values 
            
    return Xi

    
def soft_scaler(
    X,
    y =None, *, 
    kind= StandardScaler, 
    copy =True, 
    with_mean = True, 
    with_std= True , 
    feature_range =(0 , 1), 
    clip = False,
    norm ='l2',  
    **fit_params  
    ): 
    """ Quick data scaling using both strategies implemented in scikit-learn 
    with StandardScaler and MinMaxScaler. 
    
    Function returns scaled frame if dataframe is passed or ndarray. For other 
    scaling, call scikit-learn instead. 
    
    Parameters 
    ------------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to compute the mean and standard deviation
        used for later scaling along the features axis.

    y : None
        Ignored.
        
    kind: str, default='StandardScaler' 
        Kind of data scaling. Can also be ['MinMaxScaler', 'Normalizer']. The 
        default is 'StandardScaler'
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
        
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
        
    fit_params: dict, 
        keywords arguments passed to the scikit-learn fitting parameters 
        More details on https://scikit-learn.org/stable/ 
            
    Returns
    -------
    X_sc : {ndarray, sparse matrix} or dataframe of  shape \
        (n_samples, n_features)
        Transformed array.
        
    Examples 
    ----------
    >>> import numpy as np  
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import naive_scaler 
    >>> X= np.random.randn (7 , 3 ) 
    >>> X_std = naive_scaler (X ) 
    ... array([[ 0.17439644,  1.55683005,  0.24115109],
           [-0.59738672,  1.3166854 ,  1.23748004],
           [-1.6815365 , -1.19775838,  0.71381357],
           [-0.1518278 , -0.32063059, -0.47483155],
           [-0.41335886,  0.13880519,  0.69258621],
           [ 1.45221902, -1.03852015, -0.40157981],
           [ 1.21749443, -0.45541153, -2.00861955]])
    >>> # use dataframe 
    >>> Xdf = pd.DataFrame (X, columns =['a', 'c', 'c'])
    >>> naive_scaler (Xdf , kind='Normalizer') # return data frame 
    ...           a         c         c
        0  0.252789  0.967481 -0.008858
        1 -0.265161  0.908862  0.321961
        2 -0.899863 -0.416231  0.130380
        3  0.178203  0.039443 -0.983203
        4 -0.418487  0.800306  0.429394
        5  0.933933 -0.309016 -0.179661
        6  0.795234 -0.051054 -0.604150
    """
    msg =("Supports only the 'standardization','normalization' and  'minmax'"
          " scaling types, not {!r}")
    
    kind = kind or 'standard'
    
    if   ( 
            str(kind).lower().strip().find ('standard')>=0 
            or get_estimator_name(kind) =='StandardScaler'
            ): 
        kind = 'standard'
    elif ( 
            str(kind).lower().strip().find ('minmax')>=0 
            or get_estimator_name (kind) =='MinMaxScaler'
            ): 
        kind = 'minmax'
    elif  ( 
            str(kind).lower().strip().find ('norm')>=0  
            or get_estimator_name(kind)=='Normalizer'
            ):
        kind ='norm'
        
    assert kind in {"standard", 'minmax', 'norm'} , msg.format(kind)
    
    if kind =='standard': 
        sc = StandardScaler(
            copy=copy, with_mean= with_mean , with_std= with_std ) 
    elif kind == 'minmax': 
        sc = MinMaxScaler(feature_range= feature_range, 
                          clip = clip, copy =copy  ) 
    elif kind=='norm': 
        
        sc = Normalizer(copy= copy , norm = norm ) 
        
    X_sc = sc.fit_transform (X, y=y, **fit_params)
    
    if hasattr (sc , 'feature_names_in_'): 
        X_sc = pd.DataFrame( X_sc , columns = sc.feature_names_in_)  
    return X_sc 

    











        
        
        
        
        
        
        

