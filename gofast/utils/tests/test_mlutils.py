# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:13:50 2023

@author: Daniel
"""
import warnings 
import numpy as np 
import pandas as pd 
from sklearn.svm import LinearSVC, SVC 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV 

from gofast.utils.mlutils import ( 
    evaluate_model,
    select_features, 
    get_global_score,  
    get_correlated_features, 
    find_features_in, 
    codify_variables, 
    categorize_target, 
    resampling, 
    bin_counting, 
    labels_validator, 
    projection_validator, 
    rename_labels_in , 
    soft_imputer, 
    soft_scaler, 
    select_feature_importances, 
    make_pipe, 
    bi_selector, 
    get_target, 
    export_target,  
    stats_from_prediction, 
    fetch_tgz, 
    fetchModel, 
    fetch_model, 
    load_data, 
    split_train_test_by_id, 
    split_train_test, 
    discretize_categories, 
    stratify_categories, 
    serialize_data, 
    load_dumped_data, 
    naive_data_split,
    ) 
from gofast.utils.funcutils import smart_label_classifier, cleaner 
from gofast.datasets.dload import load_bagoue
from gofast.datasets.dload import load_hlogs

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
    print(yc.unique()) 
    # let visualize the number of counts 
    print( np.unique (yc, return_counts=True )) 
    data_imputed = soft_imputer(cleaned_data,  mode= 'bi-impute')
    num_scaled = soft_scaler (data_imputed[num_features],) 
    # print(num_scaled)
    # print(cleaned_data)
    # we can rencoded the target data from `make_naive_pipe` as 
    Xenc, yenc= make_pipe ( cleaned_data, y = yc ,  transform=True )
    print( np.unique (yenc, return_counts= True) ) 
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

# test functions 
def test_evaluate_model (): 
    # test the model 
    ypred, score = evaluate_model(
        model= LinearSVC(),
        X = X_train,
        y=y_train, 
        Xt= X_test,
        yt=y_test,
        eval=True, 
        )
    print("prediction", ypred) 
    print("score", score )
    
def test_select_features (): 
    X, _= _prepare_dataset(return_raw= True ) 
    
    Xcat = select_features(X, exclude='number')
    print(Xcat.head(2))
    Xnum = select_features(  X, include="number") 
    print( Xnum.head(2)) 
    print(X.columns)
    Xs= select_features (X, features = 'ohmS num shape geol lwi', 
          parse_features =True ) 
    print(Xs.columns )

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
    # return ('mean_test_score' , 'std_test_score') 
    print( get_global_score ( cvres = rd.cv_results_, )) 
   
    # (0.7396228070175438, 0.032137723656755705)
    
def test_get_correlated_features(): 
    X, _= _prepare_dataset(return_raw= True ) 
    Xnum, Xcat = bi_selector(X, return_frames =True  )
    df_spearman= get_correlated_features (
        Xcat, corr='spearman',fmt=None, threshold=.52
                   )
    print(df_spearman )
    # pearson by default
    df_pearson= get_correlated_features (Xnum,  fmt=None, threshold=.52)
    print( df_pearson ) 
    
def test_find_features_in (): 
    
    X, _= _prepare_dataset(return_raw= True ) 
    cat, num = find_features_in(X)
    print(X.columns)
    print(cat, num)
    # ... (['type', 'geol', 'shape', 'name'],
     # ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    cat, num = find_features_in(
        X, features = ['geol', 'ohmS', 'sfi'])
    print(cat, num)
    # ... (['geol'], ['ohmS', 'sfi'])
    
def test_bi_selector(): 
    data = load_hlogs().frame # get the frame 
    print(data.columns )
    # Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
    #        'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
    #        'short_distance_gamma', 'well_diameter', 'aquifer_group',
    #        'pumping_level', 'aquifer_thickness', 'hole_depth_before_pumping',
    #        'hole_depth_after_pumping', 'hole_depth_loss', 'depth_starting_pumping',
    #        'pumping_depth_at_the_end', 'pumping_depth', 'section_aperture', 'k',
    #        'kp', 'r', 'rp', 'remark'],
    #       dtype='object')
    num_features, cat_features = bi_selector (data)
    print(num_features)
    # ...['gamma_gamma',
    #      'depth_top',
    #      'aquifer_thickness',
    #      'pumping_depth_at_the_end',
    #      'section_aperture',
    #      'remark',
    #      'depth_starting_pumping',
    #      'hole_depth_before_pumping',
    #      'rp',
    #      'hole_depth_after_pumping',
    #      'hole_depth_loss',
    #      'depth_bottom',
    #      'sp',
    #      'pumping_depth',
    #      'kp',
    #      'resistivity',
    #      'short_distance_gamma',
    #      'r',
    #      'natural_gamma',
    #      'layer_thickness',
    #      'k',
    #      'well_diameter']
    print( cat_features )
    # ... ['hole_id', 'strata_name', 'rock_name', 'aquifer_group', 
    #      'pumping_level']
    
    
def test_categorize_data (): 
    def binfunc(v): 
        if v < 3 : return 0 
        else : return 1 
    arr = np.arange (10 )
    print(arr )
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    target = categorize_target(arr, func =binfunc)
    print(target)
    # array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    print(categorize_target(arr, labels =3 ))
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    # array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    print(categorize_target(arr, labels =3 , order =None ))
    # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    print(categorize_target(arr[::-1], labels =3 , order =None ))
    # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    print(categorize_target(arr, labels =[0 , 2,  4]  ))
    # array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])
    
def test_resampling ( ): 
    try : 
        # needs to instal imbalanced  
        print(X.shape, y.shape)
        data_us, target_us = resampling (X, y, kind ='under',verbose=True)
        print(data_us.shape, target_us.shape )
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
    print( Xnew.head(2) )
    # Out[7]: 
    #       power  magnitude       sfi      ohmS       lwi  shape  type  geol
    # 0  0.191800  -0.140799 -0.426916  0.386121  0.638622    4.0   1.0   3.0
    # 1 -0.430644  -0.114022  1.678541 -0.185662 -0.063900    3.0   2.0   2.0
    bc =bin_counting (Xnew , bin_columns= 'geol', tname =y).head(2)
    print(bc)
    # Out[8]: 
    #       power  magnitude       sfi      ohmS  ...  shape  type      geol  bin_target
    # 0  0.191800  -0.140799 -0.426916  0.386121  ...    4.0   1.0  0.656716           1
    # 1 -0.430644  -0.114022  1.678541 -0.185662  ...    3.0   2.0  0.219251           0
    # [2 rows x 9 columns]
    bc = bin_counting (Xnew , bin_columns= ['geol', 'shape', 'type'], tname =y).head(2)
    print(bc)
    # Out[10]: 
    #       power  magnitude       sfi  ...      type      geol  bin_target
    # 0  0.191800  -0.140799 -0.426916  ...  0.267241  0.656716           1
    # 1 -0.430644  -0.114022  1.678541  ...  0.385965  0.219251           0
    # [2 rows x 9 columns]
    df = pd.DataFrame ( pd.concat ( [Xnew, pd.Series ( y, name ='flow')],
                                       axis =1))
    bc =bin_counting (df , bin_columns= ['geol', 'shape', 'type'], 
                      tname ="flow", tolog=True).head(2)
    print(bc) 
    #       power  magnitude       sfi      ohmS  ...     shape      type      geol  flow
    # 0  0.191800  -0.140799 -0.426916  0.386121  ...  0.828571  0.364706  1.913043     1
    # 1 -0.430644  -0.114022  1.678541 -0.185662  ...  0.364865  0.628571  0.280822     0
    bc =bin_counting (df , bin_columns= ['geol', 'shape', 'type'], odds ="N-", 
                      tname =y, tolog=True).head(2)
    print(bc)
    #       power  magnitude       sfi  ...      geol  flow  bin_target
    # 0  0.191800  -0.140799 -0.426916  ...  0.522727     1           1
    # 1 -0.430644  -0.114022  1.678541  ...  3.560976     0           0
    # [2 rows x 10 columns]
    bc=bin_counting (df , bin_columns= "geol",tname ="flow", tolog=True,
                      return_counts= True )
          
    print(bc) 
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

if __name__=='__main__': 
    # test_evaluate_model()
    # test_select_features() 
    # test_get_global_score()
    # test_get_correlated_features()
    # test_find_features_in()
    # test_bi_selector () 
    # test_categorize_data () 
    # test_resampling()
    # test_bin_counting ()
    doc = store_data ()
