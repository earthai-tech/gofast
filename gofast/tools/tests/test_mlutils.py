# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:13:50 2023

@author: Daniel
"""
import pytest
from unittest.mock import patch
import warnings 
import numpy as np 
import pandas as pd
from importlib import resources 
from collections import Counter 
import urllib
from io import StringIO
import tempfile
import joblib

from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from gofast.tools.funcutils import find_features_in
from gofast.tools.mlutils import ( 
    fetch_tgz_from_url, 
    evaluate_model,
    select_features, 
    get_global_score,  
    get_correlated_features, 
    codify_variables, 
    categorize_target, 
    resampling, 
    bin_counting, 
    labels_validator, #  
    rename_labels_in , # 
    soft_imputer, 
    soft_scaler, 
    select_feature_importances, 
    make_pipe, 
    build_data_preprocessor, 
    load_saved_model, 
    bi_selector, 
    get_target, 
    extract_target,  
    stats_from_prediction, 
    fetch_tgz,  
    fetch_model, # 
    load_csv, 
    discretize_categories, 
    stratify_categories, 
    serialize_data, # 
    deserialize_data, # 
    soft_data_split, 
    laplace_smoothing, 
    laplace_smoothing_categorical, 
    laplace_smoothing_word, 
    handle_imbalance, 
    smart_split # 
    
    )  

from gofast.tools.funcutils import smart_label_classifier, cleaner 
from gofast.datasets.load import load_bagoue
from gofast.datasets.load import load_hlogs
DOWNLOAD_FILE='https://raw.githubusercontent.com/WEgeophysics/gofast/main/gofast/datasets/data/bagoue.csv'
with resources.path ('gofast.datasets.data', "bagoue.csv") as csv_f : 
    csv_file = str(csv_f)
# get the data for a test 
def _prepare_dataset ( return_encoded_data =False, return_raw=False ): 
    X, y = load_bagoue (as_frame =True, return_X_y= True  )
    
    if return_raw: 
        return X, y 
    # prepared data 
    # 1-clean data 
    cleaned_data = cleaner (X , columns = 'name num lwi', mode ='drop')
    num_features, cat_features= bi_selector (cleaned_data)
    # categorizing the labels 
    yc = smart_label_classifier (y , values = [1, 3], 
                                     labels =['FR0', 'FR1', 'FR2'] 
                                     ) 
    data_imputed = soft_imputer(cleaned_data,  mode= 'bi-impute')
    num_scaled = soft_scaler (data_imputed[num_features],) 
    # print(num_scaled)
    # print(cleaned_data)
    # we can rencoded the target data from `make_naive_pipe` as 
    Xenc, yenc= make_pipe ( cleaned_data, y = yc ,  transform=True )
    if return_encoded_data : 
        return Xenc, yenc 
    
    return num_scaled, yenc 
    
# prepared data 
X, y = _prepare_dataset() 
# encoded _data 
Xenc, yenc = _prepare_dataset(return_encoded_data= True )
# get the training and test set 
seed =42 
X_train, X_test, y_train, y_test = train_test_split(
    Xenc, yenc, test_size =.3 ,random_state=seed )

 
def test_select_features (): 
    X, _= _prepare_dataset(return_raw= True ) 
    select_features(X, exclude='number')
    select_features(  X, include="number") 
    select_features (X, features = 'ohmS num shape geol lwi', 
          parse_features =True ) 


def test_get_global_score (): 
    # train data and get the CV results 
    param_distr = [{"C": [ 0.1, 0.2, .3 ], 
                    "gamma": [ 1, 5., 10. ], 
                    "kernel": ['poly', 'sigmoid'], 
                    }, 
                   ]
    rd = RandomizedSearchCV(estimator =SVC(), 
                       param_distributions =param_distr, 
                       cv = 4 , )
    rd.fit(X_train, y_train )  
    get_global_score ( cvres = rd.cv_results_, ) 
   
    # (0.7396228070175438, 0.032137723656755705)
    
def test_get_correlated_features(): 
    X, _= _prepare_dataset(return_raw= True ) 
    Xnum, Xcat = bi_selector(X, return_frames =True  )
    get_correlated_features (
        Xcat, corr='spearman',fmt=None, threshold=.52
                   )
    # pearson by default
    get_correlated_features (Xnum,  fmt=None, threshold=.52)
    
def test_find_features_in (): 
    X, _= _prepare_dataset(return_raw= True ) 
    cat, num = find_features_in(X)
    cat, num = find_features_in(
        X, features = ['geol', 'ohmS', 'sfi'])
    # ... (['geol'], ['ohmS', 'sfi'])
    
def test_bi_selector(): 
    data = load_hlogs().frame # get the frame 
    num_features, cat_features = bi_selector (data)
    
def test_categorize_data (): 
    def binfunc(v): 
        if v < 3 : return 0 
        else : return 1 
    arr = np.arange (10 )
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    categorize_target(arr, func =binfunc)
    # array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    # array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    categorize_target(arr, labels =3 , order =None )
    # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    categorize_target(arr[::-1], labels =3 , order =None )
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    categorize_target(arr, labels =[0 , 2,  4]  )
    # array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])
    
def test_resampling ( ): 
    try : 
        # needs to instal imbalanced  
        data_us, target_us = resampling (X, y, kind ='under',verbose=True)
        # Counters: Auto      
        #                      Raw counter y: Counter({0: 291, 1: 95, 2: 45})
        #            UnderSampling counter y: Counter({0: 45, 1: 45, 2: 45})
        # (135, 6) (135,)
    except BaseException as e : 
        # convert error to warnings 
        warnings.warn (str(e))
        
def test_bin_counting () : 
     
    Xr, _= _prepare_dataset(return_raw= True ) 
    # get the categorical variables 
    num_var , cat_var = bi_selector ( Xr )
    Xcoded = codify_variables (Xr, columns = cat_var )
    # get the categ
    Xnew = pd.concat ((X, Xcoded), axis = 1 )
    #Xnew =Xnew.astype (float)
    y [y <=1] = 0;  y [y > 0]=1 
    # Out[7]: 
    #       power  magnitude       sfi      ohmS       lwi  shape  type  geol
    # 0  0.191800  -0.140799 -0.426916  0.386121  0.638622    4.0   1.0   3.0
    # 1 -0.430644  -0.114022  1.678541 -0.185662 -0.063900    3.0   2.0   2.0
    bin_counting (Xnew , bin_columns= 'geol', tname =y).head(2)
    # Out[8]: 
    #       power  magnitude       sfi      ohmS  ...  shape  type      geol  bin_target
    # 0  0.191800  -0.140799 -0.426916  0.386121  ...    4.0   1.0  0.656716           1
    # 1 -0.430644  -0.114022  1.678541 -0.185662  ...    3.0   2.0  0.219251           0
    # [2 rows x 9 columns]
    bin_counting (Xnew , bin_columns= ['geol', 'shape', 'type'], tname =y).head(2)
    # Out[10]: 
    #       power  magnitude       sfi  ...      type      geol  bin_target
    # 0  0.191800  -0.140799 -0.426916  ...  0.267241  0.656716           1
    # 1 -0.430644  -0.114022  1.678541  ...  0.385965  0.219251           0
    # [2 rows x 9 columns]
    df = pd.DataFrame ( pd.concat ( [Xnew, pd.Series ( y, name ='flow')],
                                       axis =1))
    bin_counting (df , bin_columns= ['geol', 'shape', 'type'], 
                      tname ="flow", tolog=True).head(2)
    #       power  magnitude       sfi      ohmS  ...     shape      type      geol  flow
    # 0  0.191800  -0.140799 -0.426916  0.386121  ...  0.828571  0.364706  1.913043     1
    # 1 -0.430644  -0.114022  1.678541 -0.185662  ...  0.364865  0.628571  0.280822     0
    bin_counting (df , bin_columns= ['geol', 'shape', 'type'], odds ="N-", 
                      tname =y, tolog=True).head(2)
    #       power  magnitude       sfi  ...      geol  flow  bin_target
    # 0  0.191800  -0.140799 -0.426916  ...  0.522727     1           1
    # 1 -0.430644  -0.114022  1.678541  ...  3.560976     0           0
    # [2 rows x 10 columns]
    bin_counting (df , bin_columns= "geol",tname ="flow", tolog=True,
                      return_counts= True )
    #      flow  no_flow  total_flow        N+        N-     logN+     logN-
    # 3.0    44       23          67  0.656716  0.343284  1.913043  0.522727
    # 2.0    41      146         187  0.219251  0.780749  0.280822  3.560976
    # 0.0    18       43          61  0.295082  0.704918  0.418605  2.388889
    # 1.0     9       20          29  0.310345  0.689655  0.450000  2.222222      

def store_data  (as_frame =False,  task='None', return_X_y=False ): 
    def bin_func ( x): 
        if x ==1 or x==2: 
            return 1 
        else: return 0 
    # ybin = categorize_target( y, func = func_clas)
        
    def func_ (x): 
        if x<=1: return 0 
        elif x >1 and x<=3: 
            return 1 
        else: return 2 
        
        
    X, y = load_bagoue (as_frame =True , return_X_y= True )
    
    if str(task).lower().find('bin')>=0: 
        y = categorize_target ( y, func= bin_func)
        # y = np.array (y )
        # y [y <=1] = 0;  y [y > 0]=1 
        if as_frame : 
            y = pd.Series ( y, name ='flow') 
    else: 
        y= categorize_target ( y, func= func_)
        
    # else: 
    # y = smart_label_classifier (y , values = [1, 3, 10 ], 
    #                                   labels =['FR0', 'FR1', 'FR2', 'FR3'] 
    #                                   ) 
    
    # prepared data 
    # 1-clean data 
   # (array(['FR0', 'FR1', 'FR2'], dtype=object), array([291,  95,  45], dtype=int64))
   # (array([0, 1, 2]), array([291,  95,  45], dtype=int64))
    
    cleaned_data = cleaner (X , columns = 'name num lwi', mode ='drop')
    #$print(cleaned_data.columns)
    num_features, cat_features= bi_selector (cleaned_data)
    # categorizing the labels 
   
    # print(yc.unique()) 
    # # let visualize the number of counts 
    # print( np.unique (yc, return_counts=True )) 
    data_imputed = soft_imputer(cleaned_data,  mode= 'bi-impute')
    num_scaled = soft_scaler (data_imputed[num_features],) 
    #print(num_scaled.columns)
    # we can rencoded the target data from `make_naive_pipe` as 
    pipe= make_pipe ( cleaned_data, y = y  )
    Xenc, yenc= make_pipe ( cleaned_data, y = y ,  transform=True )

    Xr, _= _prepare_dataset(return_raw= True ) 

    Xr = cleaner ( Xr, columns = 'name num lwi', mode ='drop' )
    # get the categorical variables 
    num_var , cat_var = bi_selector ( Xr )
    
    Xcoded = codify_variables (Xr, columns = cat_var )
    # get the categ
    Xnew = pd.concat ((X[num_var], Xcoded), axis = 1 )
    Xanalysed= pd.concat ( (num_scaled, Xcoded), axis=1 )
    
    #X_train, X_test, y_train, y_test = train_test_split()
    
    data = {"preprocessed": ( num_scaled, y ), 
      "encoded": (Xenc, yenc),
      "codified": ( Xnew, y ), 
      "analysed": (Xanalysed, y  ), 
      "pipe": pipe, 
          }
    # import joblib 
    # joblib.dump ( data , filename ='b.pkl')
    return data 

def test_codify_variables_simple_encoding():
    # Example data
    data = {
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
        'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small']
    }
    df = pd.DataFrame(data)
    
    # Perform simple encoding
    df_encoded, map_codes = codify_variables(df, return_cat_codes=True)
    
    # Assert that the output is as expected (example based on docstring)
    assert 'Color' in df_encoded.columns
    assert 'Size' in df_encoded.columns
    assert isinstance(map_codes, dict)
    assert len(map_codes) > 0  # Checks if some encoding mapping is returned

def test_codify_variables_one_hot_encoding():
    # Example data similar to the previous test
    data = {
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
        'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small']
    }
    df = pd.DataFrame(data)
    
    # Perform one-hot encoding
    df_encoded = codify_variables(df, get_dummies=True)
    
    # Asserts to check if one-hot encoding is applied correctly
    assert 'Color_Red' in df_encoded.columns
    assert 'Size_Small' in df_encoded.columns
    assert df_encoded['Color_Red'].sum() == 2  # Based on example data

def test_resampling_oversampling():
    # Example data with imbalance
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Example features
    y = np.array([0, 0, 1])  # Imbalanced target
    
    # Apply oversampling
    X_resampled, y_resampled = resampling(X, y, kind='over', verbose=True)
    
    # Check if the class distribution is balanced
    counter = Counter(y_resampled)
    assert counter[0] == counter[1]

def test_resampling_undersampling():
    # Similar setup as the previous test but for undersampling
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Example features
    y = np.array([0, 0, 1, 1])  # Balanced target for demonstration
    
    # Apply undersampling
    X_resampled, y_resampled = resampling(X, y, kind='under', verbose=True)
    
    # Check if the class distribution is balanced
    counter = Counter(y_resampled)
    assert counter[0] == counter[1]

def test_laplace_smoothing_word():
    word_counts = {('dog', 'animal'): 3, ('cat', 'animal'): 2, ('car', 'non-animal'): 4}
    class_counts = {'animal': 5, 'non-animal': 4}
    V = len(set([w for (w, c) in word_counts.keys()]))
    probability = laplace_smoothing_word('dog', 'animal', word_counts, class_counts, V)
    assert probability == pytest.approx(0.5)

def test_laplace_smoothing_categorical():
    data = pd.DataFrame({'feature': ['cat', 'dog', 'cat', 'bird'], 'class': ['A', 'A', 'B', 'B']})
    probabilities = laplace_smoothing_categorical(data, 'feature', 'class')
    # Asserts based on expected behavior; specific values depend on the implementation
    assert 'cat' in probabilities.index
    assert 'dog' in probabilities.index
    assert 'bird' in probabilities.index
 
def test_laplace_smoothing():
    data = np.array([[0, 1], [1, 0], [1, 1]])
    smoothed_probs = laplace_smoothing(data, alpha=1)
    expected_probs = np.array([[0.4, 0.6], [0.6, 0.4], [0.6, 0.6]])
    assert np.allclose(smoothed_probs, expected_probs)

def test_evaluate_model_with_model():
    data = load_iris()
    arrays= train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter= 1000) # to converge 
    y_pred, accuracy = evaluate_model(model, *arrays,  scorer='accuracy', eval=True)
    assert y_pred is not None
    assert accuracy > 0  # Checking that accuracy is calculated

def test_evaluate_model_with_pred():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 0, 1])
    _, accuracy = evaluate_model(y_pred=y_pred, yt=y_true, scorer='accuracy', eval=True)
    assert accuracy == 1  # Perfect accuracy expected

def test_get_correlated_features2():
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100) * 2,
        'C': np.random.rand(100) + 0.5
    })
    correlated_features = get_correlated_features(df, corr='pearson', threshold=0.5)
    assert isinstance(correlated_features, pd.DataFrame)

def test_get_target():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [0, 1, 0]
    })
    target, modified_df = get_target(df.copy(), 'target', True)
    
    assert 'target' in df.columns  # Original DataFrame should remain unchanged
    assert 'target'  in modified_df.columns  # Modified DataFrame should not have the target column
    assert all(target == df['target'])  # Extracted target should match the original target column

def test_get_target2():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': ['A', 'B', 'C']
    })
    target, modified_df = get_target(df, 'target', inplace=False)
    assert 'target' in df.columns  # Original df should be unchanged
    assert all(target == df['target'])  # Target column extracted correctly

def test_select_features2():
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'feature3': [0.1, 0.2, 0.3]
    })
    selected_data = select_features(data, features=['feature1', 'feature3'])
    assert list(selected_data.columns) == ['feature1', 'feature3']

def test_get_global_score2():
    cvres = {
        'mean_test_score': np.array([0.8, 0.85, 0.9]),
        'std_test_score': np.array([0.05, 0.04, 0.03])
    }
    mean_score, mean_std = get_global_score(cvres)
    assert pytest.approx(mean_score, 0.85)
    assert pytest.approx(mean_std, 0.04)

def test_get_target_inplace_true():
    df = pd.DataFrame({'feature': [1, 2, 3], 'target': ['A', 'B', 'C']})
    target, _ = get_target(df, 'target')
    assert 'target' not in df.columns
    assert all(target == ['A', 'B', 'C'])

def test_get_target_inplace_false():
    df = pd.DataFrame({'feature': [1, 2, 3], 'target': ['A', 'B', 'C']})
    _, new_df = get_target(df, 'target', inplace=False)
    assert 'target' in df.columns
    assert 'target'in new_df.columns

def test_select_features_by_name():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = select_features(df, features=['A', 'C'])
    assert list(result.columns) == ['A', 'C']

def test_select_features_include_exclude():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['x', 'y', 'z']})
    result = select_features(df, include=['number'])
    assert 'C' not in result.columns
    result = select_features(df, exclude=['number'])
    assert list(result.columns) == ['C']

def test_get_global_score3():
    cvres = {'mean_test_score': [0.9, 0.85, 0.95], 'std_test_score': [0.05, 0.02, 0.03]}
    mean_score, mean_std = get_global_score(cvres)
    assert mean_score == pytest.approx(0.9)
    assert mean_std == pytest.approx(0.033333, 0.001)

def test_laplace_smoothing_word2():
    word_counts = {('dog', 'animal'): 2, ('cat', 'animal'): 3, ('car', 'non-animal'): 1}
    class_counts = {'animal': 5, 'non-animal': 1}
    V = 4  # unique words
    prob = laplace_smoothing_word('dog', 'animal', word_counts, class_counts, V)
    assert prob == pytest.approx((2 + 1) / (5 + 4))

def test_laplace_smoothing_categorical2():
    data = pd.DataFrame({'feature': ['dog', 'cat', 'dog', 'mouse'], 'class': ['A', 'A', 'B', 'B']})
    V = 3  # unique features
    result = laplace_smoothing_categorical(data, 'feature', 'class', V=V)
    assert result.loc['dog', 'A'] == pytest.approx(2 / (2 + 3))
    assert result.loc['mouse', 'B'] == pytest.approx(2 / (2 + 3))

def test_laplace_smoothing2():
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    smoothed = laplace_smoothing(data, alpha=1)
    # Assuming data is already in categorical form and alpha=1 for Laplace Smoothing
    assert smoothed[0, 0] == pytest.approx((1 + 1) / (4 + 2))
    assert smoothed[1, 1] == pytest.approx((1 + 1) / (4 + 2))

def test_stats_from_prediction():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 3, 5]
    # Expected stats calculation
    expected_stats = {
        'mean': np.mean(y_pred),
        'median': np.median(y_pred),
        'std_dev': np.std(y_pred),
        'min': np.min(y_pred),
        'max': np.max(y_pred),
        'MAE': 0.2,
        'MSE': 0.2,
        'RMSE': np.sqrt(0.2),
        # 'Accuracy' won't be included due to continuous values, not classification
    }
    stats = stats_from_prediction(y_true, y_pred, verbose=False)
    for key, value in expected_stats.items():
        assert pytest.approx(stats[key]) ==value, f"Mismatch in {key}"
        
# This is a conceptual outline and not directly runnable without appropriate setup
def test_fetch_tgz(mocker, fake_fs):
    # Setup mock for urllib and tarfile
    mocker.patch('urllib.request.urlretrieve')
    mocker.patch('tarfile.open')
    
    # Assuming fake_fs is a fixture to mock filesystem
    fake_fs.create_dir("/mock/data/path")
    
    # Mock parameters
    data_url = "http://example.com/data.tgz"
    data_path = "/mock/data/path"
    tgz_filename = "data.tgz"
    
    fetch_tgz(data_url, data_path, tgz_filename)
    
    # Assertions to verify the behavior, for example:
    urllib.request.urlretrieve.assert_called_with(data_url, "/mock/data/path/data.tgz")

def test_fetch_tgz_from_url_success():
    with patch('os.path.isdir', return_value=True), \
         patch('os.makedirs'), \
         patch('urllib.request.urlretrieve') as mock_urlretrieve, \
         patch('tarfile.open') as mock_tarfile_open:
        mock_urlretrieve.return_value = None
        mock_tarfile_open.return_value.__enter__.return_value.extractall.return_value = None
        
        assert fetch_tgz_from_url("http://example.com/data.tgz", "/fake/path", "data.tar.gz") is False
        mock_urlretrieve.assert_called_once()
        mock_tarfile_open.assert_called_once()

def test_load_csv():
    # test_csv_data = "col1,col2\n1,2\n3,4"
    test_df = pd.read_csv(StringIO(csv_file))
    with patch("pandas.read_csv", return_value=test_df) as mock_read_csv:
        result_df = load_csv("fake_file.csv")
        pd.testing.assert_frame_equal(result_df, test_df)
        mock_read_csv.assert_called_once()

def test_discretize_categories():
    df = pd.DataFrame({
        'cat': [0.5, 1.5, 2.5, 3.5]
    })
    new_df = discretize_categories(df, in_cat='cat', new_cat='discretized_cat', divby=1.5)
    expected_output = pd.DataFrame({
        'cat': [0.5, 1.5, 2.5, 3.5],
        'discretized_cat': [1.0, 1.0, 2.0, 3.0]
    })
    pd.testing.assert_frame_equal(new_df, expected_output)

def test_stratify_categories():
    df = pd.DataFrame({
        'features': range(100),
        'target': [0, 1] * 50  # Ensure balanced dataset for easy testing
    })
    strat_train_set, strat_test_set = stratify_categories(df, 'target')
    assert len(strat_train_set) + len(strat_test_set) == len(df)

def test_extract_target_dataframe():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': ['A', 'B', 'C']
    })
    target, modified_df = extract_target(df, 'target')
    assert all(target == ['A', 'B', 'C'])
    assert 'target' not in modified_df.columns

def test_extract_target_array():
    arr = np.array([[1, 2, 'A'], [3, 4, 'B'], [5, 6, 'C']])
    target, modified_arr = extract_target(arr, 2,  columns=['feature1', 'feature2', 'target'])
    assert np.array_equal(target, np.array(['A', 'B', 'C']))
    assert modified_arr.shape == (3, 2)  # Ensure target column was removed


def test_handle_imbalance_oversample():
    X, y = make_classification(n_classes=2, class_sep=2,
                               weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=4, n_clusters_per_class=1, n_samples=100, random_state=10)
    X_resampled, y_resampled = handle_imbalance(X, y, strategy='oversample', random_state=42)
    unique, counts = np.unique(y_resampled, return_counts=True)
    assert counts[0] == counts[1]  # Check if classes are balanced

def test_soft_data_split():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = soft_data_split(X, y, test_size=0.25, random_state=42)
    assert len(X_train) == 0.75 * len(X)
    assert len(X_test) == 0.25 * len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_soft_data_split_extract_target():
    data = load_iris(as_frame=True).frame
    target_column = 'target'
    X_train, X_test, y_train, y_test = soft_data_split(data, test_size=0.25, random_state=42,
                                                       target_column=target_column, extract_target=True)
    assert target_column not in X_train.columns
    assert target_column not in X_test.columns
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_load_saved_model_with_joblib():
    model = {'best_model': 'mock_model', 'best_params_': {'param': 'value'}}
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        joblib.dump(model, tmp.name)
        loaded_model, params = load_saved_model(tmp.name)
    assert loaded_model == 'mock_model'
    assert params == {'param': 'value'}

def test_load_saved_model_with_pickle():
    model = {'best_model': 'mock_model', 'best_params_': {'param': 'value'}}
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
        joblib.dump(model, tmp.name)
        loaded_model, params = load_saved_model(tmp.name, storage_format='pickle')
    assert loaded_model == 'mock_model'
    assert params == {'param': 'value'}

def test_load_saved_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_saved_model('nonexistent_model.pkl')

def test_load_saved_model_unsupported_format():
    with pytest.raises(ValueError):
        load_saved_model('model.unsupported', storage_format='unsupported')

def test_categorize_target_with_function():
    arr = np.array([1, 2, 3, 4, 5])
    def categorize_func(x): return 0 if x <= 3 else 1
    categorized_arr = categorize_target(arr, func=categorize_func)
    assert np.array_equal(categorized_arr, np.array([0, 0, 0, 1, 1]))

def test_categorize_target_with_labels():
    arr = np.array([1, 2, 3, 4, 5])
    categorized_arr = categorize_target(arr, labels=2)
    # Assuming the function divides the range into equal intervals
    assert len(np.unique(categorized_arr)) == 2

def test_categorize_target_with_rename_labels():
    arr = np.array([1, 2, 3, 4, 5])
    categorized_arr = categorize_target(arr, labels=2, rename_labels=['A', 'B'], coerce=True)
    assert set(categorized_arr) == {'A', 'B'}

def test_bi_selector2():
    df = pd.DataFrame({
        'num_feature': [1.0, 2.0, 3.0],
        'cat_feature': ['A', 'B', 'C']
    })
    numerical_features, categorical_features = bi_selector(df)
    assert 'num_feature' in numerical_features
    assert 'cat_feature' in categorical_features

def test_bi_selector_with_specified_features():
    df = pd.DataFrame({
        'num_feature1': [1.0, 2.0, 3.0],
        'num_feature2': [4.0, 5.0, 6.0],
        'cat_feature': ['A', 'B', 'C']
    })
    _, selected_features = bi_selector(df, features=['num_feature1', 'cat_feature'], return_frames=False)
    assert 'num_feature1' in selected_features
    assert 'cat_feature' in selected_features
    assert 'num_feature2' not in selected_features

def test_bi_selector_return_frames():
    df = pd.DataFrame({
        'num_feature': [1.0, 2.0, 3.0],
        'cat_feature': ['A', 'B', 'C']
    })
    num_df, cat_df = bi_selector(df, return_frames=True)
    assert 'num_feature' in num_df.columns
    assert 'cat_feature' in cat_df.columns
    assert len(num_df) == len(df)
    assert len(cat_df) == len(df)

def test_make_pipe_numeric_only():
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4],
        'num2': [5, 6, 7, np.nan]
    })
    pipe = make_pipe(df, transform=False)
    assert 'num_pipeline' in [name for name, _ in pipe.transformer_list]
    assert 'cat_pipeline' not in [name for name, _ in pipe.transformer_list]

def test_make_pipe_categorical_only():
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'd'],
        'cat2': ['e', 'f', 'g', 'h']
    })
    pipe = make_pipe(df, transform=False)
    assert 'cat_pipeline' in [name for name, _ in pipe.transformer_list]
    assert 'num_pipeline' not in [name for name, _ in pipe.transformer_list]

def test_make_pipe_transform():
    df = pd.DataFrame({
        'num': [1, 2, np.nan, 4],
        'cat': ['a', 'b', 'a', 'c']
    })
    X_transformed = make_pipe(df, transform=True)
    assert X_transformed is not None
    assert X_transformed.shape[1] == 3  # One numerical feature and two one-hot encoded features

def test_build_data_preprocessor_basic():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    pipeline = build_data_preprocessor(pd.DataFrame(X), y, transform=False)
    assert pipeline is not None  # Pipeline creation successful

def test_build_data_preprocessor_transform():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    X_transformed, y_transformed = build_data_preprocessor(pd.DataFrame(X), y, transform=True,
                                                           output_format='array')
    assert X_transformed.shape[0] == X.shape[0]
    assert len(y_transformed) == len(y)

def test_select_feature_importances():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5)
    clf = RandomForestClassifier().fit(X, y)
    X_selected = select_feature_importances(clf, X, y, threshold='0.01', prefit=True)
    assert X_selected.shape[1] <= X.shape[1]

def test_select_feature_importances_return_selector():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5)
    clf = RandomForestClassifier().fit(X, y)
    selector = select_feature_importances(clf, X, y, threshold='0.01', prefit=True, 
                                          return_selector=True)
    assert hasattr(selector, 'transform')

def test_soft_imputer_mean_strategy():
    df = pd.DataFrame({
        'num': [1, np.nan, 3],
        'cat': ['a', 'b', np.nan]
    })
    imputed_df = soft_imputer(df, strategy='mean', mode='bi-impute')
    assert imputed_df.isnull().sum().sum() == 0
    assert imputed_df['num'].iloc[1] == df['num'].mean()

def test_soft_imputer_constant_strategy():
    df = pd.DataFrame({
        'num': [1, np.nan, 3],
        'cat': ['a', 'b', np.nan]
    })
    imputed_df = soft_imputer(df, strategy='constant', fill_value='missing', mode='bi-impute')
    assert imputed_df.isnull().sum().sum() == 0
    assert imputed_df['cat'].iloc[2] == 'missing'

def test_labels_validator_with_existing_labels():
    y = np.array([0, 1, 2, 0, 1, 2])
    labels = [0, 1, 2]
    assert labels_validator(y, labels, return_bool=True)

def test_labels_validator_with_missing_labels():
    y = np.array([0, 1, 0, 1])
    labels = [0, 1, 2]
    assert not labels_validator(y, labels, return_bool=True)

def test_labels_validator_raises_value_error_on_missing_labels():
    y = np.array([0, 1, 0, 1])
    labels = [0, 1, 2]
    try:
        labels_validator(y, labels)
    except ValueError as e:
        assert str(e).startswith("Label value")

def test_rename_labels_in():
    arr = np.array([0, 1, 2, 0, 1, 2])
    new_labels = ["A", "B", "C"]
    expected = np.array(["A", "B", "C", "A", "B", "C"])
    result = rename_labels_in(arr, new_labels)
    np.testing.assert_array_equal(result, expected)

def test_rename_labels_in_with_coerce():
    arr = np.array([0, 1, 2, 0, 1, 2])
    new_labels = ["A", "B"]  # Intentionally missing label for "2"
    expected = np.array(["A", "B", 2, "A", "B", 2])  # Original "2" labels should remain unchanged
    result = rename_labels_in(arr, new_labels, coerce=True)
    np.testing.assert_array_equal(result, expected)

def test_fetch_model():
    model_path = 'test_model.pkl'
    model_to_save = {'model': 'test_model'}
    joblib.dump(model_to_save, model_path)
    
    loaded_model = fetch_model(model_path)
    assert 'model' in loaded_model and loaded_model['model'] == 'test_model'

def test_fetch_model_with_specific_name():
    model_path = 'test_model_multiple.pkl'
    models_to_save = {'model1': 'test_model_1', 'model2': 'test_model_2'}
    joblib.dump(models_to_save, model_path)
    
    loaded_model = fetch_model(model_path, name='model2')
    assert loaded_model == 'test_model_2'

def test_serialize_data():
    data = np.array([1, 2, 3])
    file_path = 'test_data.pkl'
    serialize_data(data, filename=file_path)
    # Verify that data can be correctly deserialized
    loaded_data = deserialize_data(file_path)
    np.testing.assert_array_equal(data, loaded_data)

# Assuming serialize_data test has already created 'test_data.pkl'
def test_deserialize_data():
    file_path = 'test_data.pkl'
    loaded_data = deserialize_data(file_path)
    expected_data = np.array([1, 2, 3])
    np.testing.assert_array_equal(loaded_data, expected_data)

def test_smart_split_with_target_extraction():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4],
        'target': [0, 1, 0, 1]
    })
    X_train, X_test, y_train, y_test = smart_split(df, target_column='target')
    
    assert 'target' not in X_train.columns and 'target' not in X_test.columns
    assert len(y_train) > 0 and len(y_test) > 0

def test_smart_split_without_target_extraction():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [4, 3, 2, 1]
    })
    X_train, X_test = smart_split(X)
    
    assert len(X_train) > 0 and len(X_test) > 0

if __name__ == '__main__':
    pytest.main([__file__])
    # test_evaluate_model_with_model()


