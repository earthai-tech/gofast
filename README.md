# gofast: _Accelerate Your Machine Learning Workflow_

**gofast** is a comprehensive machine learning toolbox designed to streamline and accelerate every step of your data science workflow. This project is focused on delivering high-speed tools and utilities that assist users in swiftly navigating through the critical stages of data analysis, processing, and modeling.

## Key Features

- **Fast Preprocessing**: Simplify and expedite your data preparation with our efficient preprocessing tools. gofast ensures your data is clean, formatted, and ready for analysis in record time.

- **Efficient Processing**: Our robust processing utilities are optimized for speed without compromising accuracy. Tackle large datasets and complex computations with ease.

- **Streamlined Validation**: Quick and reliable validation methods to assess your models effectively. gofast helps in fine-tuning model performance, ensuring you get the best results.

## Project Goals

- **Enhance Productivity**: Reduce the time spent on routine data tasks. gofast is here to make your workflow more efficient, allowing you to focus on innovation and problem-solving.

- **User-Friendly**: Whether you're a beginner or an expert, gofast is designed to be intuitive and accessible for all users in the machine learning community.

- **Community-Driven**: We believe in the power of collaboration. gofast is an open-source project, welcoming contributions and suggestions from the community to continuously improve and evolve.


## Demo 

For demonstration, we will use a composite dataset stored in the software. The data is composed 
of dirty numerical and categorical variables including useless features: 

```python  
>>> import gofast as gf 
>>> X, y = gf.fetch_data ('bagoue', as_frame =True, return_X_y=True )
>>> X.head(2) 
Out[1]: 
   num name      east  ...         ohmS        lwi                    geol
0  1.0   b1  790496.0  ...  1666.165142  32.023585  VOLCANO-SEDIM. SCHISTS
1  2.0   b2  791227.0  ...  1135.551531  21.406531                GRANITES

[2 rows x 12 columns]
``` 
We will try a fast manipulation of data in a few line of codes: 

1. **Fast clean and sanitize raw data** 

``gofast`` has a capability to clean your data, strip the columns bad string characters, 
drop your useless features (e.g., `name`, `num` and `lwi`)  in a one line of code as: 

```python
>>> cleaned_data = gf.cleaner (X , columns = 'name num lwi', mode ='drop')
>>> cleaned_data.shape 
Out[2]: (431, 9)
``` 
2. **Split numerical and categorical data in one line of code**
 
User does need to care about the columns, `gofast` does it for you thanks to 
``bi_selector``function by turning the `return_frames` argument to ``True``. By 
default, it returns the distinct numerical and categorical feature names   as: 

```python 
>>> num_features, cat_features= gf.bi_selector (cleaned_data)
Out[3]: 
(['sfi', 'east', 'ohmS', 'power', 'magnitude', 'north'],
 ['shape', 'type', 'geol'])
 
```
If features are explicitly given through the argument of parameters `features`, ``gofast`` 
returns your remained features accordingly. Here is an example: 

```python 
>>> remained_features, my_features = gf.bi_selector ( cleaned_data, features ='shape ohmS')
Out[4]: 
(['geol', 'sfi', 'east', 'power', 'type', 'magnitude', 'north'],
 ['shape', 'ohmS'])
```
3. **Dual imputation** 

``gofast`` is able to impute at the same time, the numeric and categorical data 
via the ``bi-impute`` strategy  in one line of code  as 

```python 
>>> data_imputed = gf.soft_imputer(cleaned_data,  mode= 'bi-impute')
``` 
The data imputation can be controlled via the parameters `strategy`, `drop_features`, `missing_values`
`fill_value`. By default, `the most_frequent` argument is used to impute the categorical features.

4. **Automated pipeline with your data** 

``gofast`` understands your data and create a fast pipeline for you. If the data contains
missing values or too dirty, ``gofast`` sanitizes it before proceeding. Multiple lines 
of codes can henceforth be skipped. This is a proposed pipeline of ``gofast`` according to 
your dataset: 

```python 
>>> auto_pipeline = gf.make_pipe (cleaned_data )
Out[5]: 
FeatureUnion(transformer_list=[('num_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['sfi', 'east', 'ohmS', 'power', 'magnitude', 'north'])),
                                                ('imputerObj',
                                                 SimpleImputer(strategy='median')),
                                                ('scalerObj',
                                                 StandardScaler())])),
                               ('cat_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['shape', 'type', 'geol'])),
                                                ('OneHotEncoder',
                                                 OneHotEncoder())]))])
```  
Rather to return an automated pipeline, users can outputed the transformed data by setting 
`` the argument of `transform ` parameter to ``True``   as 
```python 

>>> transformed_data = gf.make_pipe ( cleaned_data, transform=True )
Out[6]: 
<431x19 sparse matrix of type '<class 'numpy.float64'>'
	with 3879 stored elements in Compressed Sparse Row format>
```
5. **Manage smartly your target**

For a classification problem, ``gofast`` can efficiently manage your target by specifying  
the class boundaries and labels names  Then ``gofast`` does it and returns categorized 
targets via the ``smart_label_classifier`` function accordingly. Here is an example
```python 
>>> import numpy as np 
>>> from sklearn.ensemble import RandomForestClassifier
>>> # categorizing the labels 
>>> yc = gf.smart_label_classifier (y , values = [1, 3, 10 ], 
                                 labels =['child', 'teenager', 'young', 'adult'] 
                                 ) 
>>> yc.unique() 
Out[7]: array(['teenager', 'child', 'young', 'adult'], dtype=object)
>>> # let visualize the number of counts 
>>> np.unique (yc, return_counts=True )
Out[8]: 
(array(['adult', 'child', 'teenager', 'young'], dtype=object),
 array([  4, 291,  95,  41], dtype=int64))
>>> # we can rencoded the target data from `make_naive_pipe` as 
>>> Xenc, yenc= gf.make_pipe ( cleaned_data, y = yc ,  transform=True )
>>> np.unique (yenc, return_counts= True) 
Out[9]: (array([0, 1, 2, 3]), array([  4, 291,  95,  41], dtype=int64))
``` 

6. **Train multiple estimators** 

Do you want to train multiple estimators at the same time? Don't worry ``gofast`` 
does it for you and save your results into a binary disk. The ``GridSearchMultiple`` 
is built to simplify your task. Here is an example:

```python 
>>> from sklearn.svm import SVC, LinearSVC 
>>> from sklearn.linear_model import SGDClassifier,LogisticRegression
>>> from gofast.validation import GridSearchMultiple, displayFineTunedResults
>>> # As example, we can build 04 estimators and provide their 
>>> # grid parameters range for fine-tuning as :
>>> random_state=42
>>> logreg_clf = LogisticRegression(random_state =random_state)
>>> linear_svc_clf = LinearSVC(random_state =random_state)
>>> sgd_clf = SGDClassifier(random_state = random_state)
>>> svc_clf = SVC(random_state =random_state) 
>>> estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
>>> grid_params= ([dict(C=[1e-2, 1e-1, 1, 10, 100], 
                        gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],kernel=['rbf']), 
                   dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3],
                        C= [1e-2, 1e-1, 1, 10, 100])],
                [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
                [dict()], # we just no provided parameter for demo
                [dict()]
                )
>>> # Now train and self-validate all  with CV=4 via the RandomizedSearchCV
>>> gobj = GridSearchMultiple(estimators = estimators, 
                       grid_params = grid_params ,
                       cv =4, 
                       scoring ='accuracy', 
                       verbose =1,   # > 7 output more messages 
                       savejob=False ,  # set true to save job in binary disk file.
                       kind='RandomizedSearchCV')
>>> gobj.fit(Xenc, yenc) # Starts the training
>>> # Once the parameters are fined tuned, we can display the fined tuning 
>>> # results using ``displayFineTunedResults`` function
>>> displayFineTunedResults (gobj.models.values_) 
Out[10]:
MODEL NAME = SVC
BEST PARAM = {{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}}
BEST ESTIMATOR = SVC(C=100, gamma=0.01, random_state=42)

MODEL NAME = LinearSVC
BEST PARAM = {{'C': 100, 'loss': 'hinge'}}
BEST ESTIMATOR = LinearSVC(C=100, loss='hinge', random_state=42)

MODEL NAME = LogisticRegression
BEST PARAM = {{}}
BEST ESTIMATOR = LogisticRegression(random_state=42)

MODEL NAME = SGDClassifier
BEST PARAM = {{}}
BEST ESTIMATOR = SGDClassifier(random_state=42)
``` 
7. **Plot feature importances** 

``gofast`` helps for a fast feature contributions visualization. Here is 
an example using the ``sklearn.ensemble.RandomForestClassifier``: 

```python 
>>> from sklearn.ensemble import RandomForestClassifier 
>>> from gofast.plot import plot_rf_feature_importances 
>>> # Try to scale the numeric data 
>>> num_scaled = gf.soft_scaler (data_imputed[num_features],)  
>>> plot_rf_feature_importances (RandomForestClassifier(), num_scaled, yenc) 
``` 

## Note 
 **gofast** is still under development, the first version should be released soon. 
 
 
## Contributions 

_Join us in making machine learning workflows faster and more efficient. With gofast, you're not just processing data; you're accelerating towards breakthroughs and innovations._
