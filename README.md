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

For demonstration, we use a composite dataset named `medical_diagnostic` stored in the software. 
The data is composed of a mixte of numerical and categorical variables with multioutput target `y` by default.
For the example, we keep the  `HistoryOfDiabetes` as a unique output target and drop the rest.

```python  
>>> import gofast as gf 
>>> from gofast.datasets import make_medical_diagnostic 
>>> X, y = make_medical_diagnostic (as_frame =True, return_X_y=True , tnames='HistoryOfDiabetes')
>>> X.head(2) 
Out[1]: 
   Age  Gender  ... Cholesterol_mg_dL  Hemoglobin_g_dL
0   57  Female  ...        213.919645        13.191829
1    6    Male  ...        163.111149        12.932728

[2 rows x 12 columns]
``` 
We will try a fast manipulation of data in a few lines of codes: 

1. **Fast clean and sanitize raw data** 

``gofast``can clean your data, strip the column bad string characters, 
and drop your useless features in one line of code. To drop  `'HistoryOfHypertension'` and
`'HistoryOfHeartDisease'`, we can process as : 

```python
>>> cleaned_data = gf.cleaner (X , columns = 'HistoryOfHypertension HistoryOfHeartDisease', mode ='drop')
>>> cleaned_data.shape 
Out[2]: (1000, 12)
``` 
2. **Split numerical and categorical data in one line of code**
 
The user does not need to care about the columns, `gofast` does it for you thanks to 
the ``bi_selector`` function by turning the `return_frames` argument to ``True``. By 
default, the function returns the distinct numerical and categorical feature names as: 

```python 
>>> num_features, cat_features= gf.bi_selector (cleaned_data, return_frames=False)
Out[3]: 
(['Hemoglobin_g_dL',
  'SystolicBP',
  'Age',
  'Temperature_C',
  'DiastolicBP',
  'BloodSugar_mg_dL',
  'Cholesterol_mg_dL',
  'HeartRate',
  'Weight_kg',
  'Height_cm'],
 ['Gender', 'Ethnicity'])
 
```
If features are explicitly passed through the argument of parameter `features`, ``gofast`` 
returns the remained features accordingly. Here is an example: 

```python 
>>> remained_features, my_features = gf.bi_selector ( cleaned_data, features ='HeartRate DiastolicBP', 
                                                     parse_features=True )
Out[4]: 
(['Ethnicity',
  'Hemoglobin_g_dL',
  'SystolicBP',
  'Age',
  'Temperature_C',
  'BloodSugar_mg_dL',
  'Cholesterol_mg_dL',
  'Gender',
  'Weight_kg',
  'Height_cm'],
 ['HeartRate', 'DiastolicBP'])
```
3. **Dual imputation** 

``gofast``can impute at the same time, the numeric and categorical data 
via the ``bi-impute`` strategy  in a one snippet code as 

```python 
>>> data_imputed = gf.soft_imputer(cleaned_data,  mode= 'bi-impute')
``` 
The data imputation can be controlled via the parameters `strategy`, `drop_features`, `missing_values`, or
`fill_value`. By default, `the most_frequent` argument is used to impute the categorical features.

4. ** Data-based automate pipeline** 

``gofast`` understands your data and creates a fast pipeline for you. If the data contains
missing values or is too dirty, ``gofast`` sanitizes it before proceeding. Multiple lines 
of codes can  be skipped henceforth. Here is a proposed pipeline of ``gofast`` based
on a medical diagnostic dataset: 
```python 
>>> auto_pipeline = gf.make_pipe (cleaned_data )
Out[5]: 
FeatureUnion(transformer_list=[('num_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['Hemoglobin_g_dL', 'SystolicBP', 'Age', 'Temperature_C', 'DiastolicBP', 'BloodSugar_mg_dL', 'Cholesterol_mg_dL', 'HeartRate', 'Weight_kg', 'Height_cm'])),
                                                ('imputerObj',
                                                 SimpleImputer(strategy='median')),
                                                ('scalerObj',
                                                 StandardScaler())])),
                               ('cat_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['Gender', 'Ethnicity'])),
                                                ('OneHotEncoder',
                                                 OneHotEncoder())]))])
```  
Rather than returning an automated pipeline, the user can get the transformed data in 
one line of code by setting the argument of the `transform` parameter to ``True`` as 
```python 

>>> transformed_data = gf.make_pipe ( cleaned_data, transform=True )
Out[6]: 
<1000x16 sparse matrix of type '<class 'numpy.float64'>'
	with 12000 stored elements in Compressed Sparse Row format>
```
``Gofast`` provides many other parameters essential for the user to control its dataset
before the data transformation. 
 
5. **Manage smartly your target**

For a classification problem, ``gofast`` can efficiently manage your target by specifying  
the class boundaries and label names. Then ``gofast`` performs operations and returns categorized 
targets thanks to the ``smart_label_classifier`` function. Here is an example
```python 
>>> import numpy as np 
>>> # categorizing the labels 
>>> np.random.seed(42) # reproduce the same target 
>>> person_ages= np.random.randint ( 1, 120 , 100 )
>>> ages_classes = gf.smart_label_classifier (person_ages , values = [10, 18, 25 ], 
                                 labels =['child', 'teenager', 'young', 'adult'] 
                                 ) 
>>> # boundaries: <=10: child; (10, 18]: teenager; (18, 25]: young and >25:adults 
>>> # let visualize the number of counts 
>>> np.unique (ages_classes, return_counts=True )
Out[7]: 
(array(['adult', 'child', 'teenager', 'young'], dtype=object),
 array([76, 13,  5,  6], dtype=int64))
``` 

6. **Train multiple estimators** 

Do you want to train multiple estimators in parallel(at the same time)? Don't worry ``gofast`` 
does it for you and saves your results into a binary disk. The ``parallelize_estimators`` 
function is built to simplify your task. To understand why ``gofast`` deserves its name, 
let's try to evaluate three estimators (`SVC`, `DecisionTreeClassifier` and `LogisticRegression`)
on a simple IRIS dataset. Then we compare the time elapsed with naive and the parallelized approach 
proposed by ``gofast``. Let's do this!
```python
>>> import time 
>>> from sklearn.svm import SVC 
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from gofast.datasets import load_iris

>>> from gofast.models.optimize import parallelize_estimators 
>>> from gofast.tools.validator import get_estimator_name
>>> # load the dataset 
>>> X, y = load_iris(return_X_y=True)
>>> # construct the estimators or classifiers in our case.
>>> estimators = [SVC(), DecisionTreeClassifier(), LogisticRegression ()]
>>> # let get the names of estimators 
>>> names =[ get_estimator_name( estimator) for estimator in estimators ]
>>> # Define grid of parameters for GridSearchCV
>>> param_grids = [{ # for SVM
                     'C': [1, 10], 'kernel': ['linear', 'rbf']
                     }, 
                   {  # for DecisionTreeClassifier
                   'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']
                   }, 
                   { # for Logistic Regression
                   'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2'],'solver': ['liblinear']
                   }]
>>> 
>>> # (1) Naive approach 
>>> 
>>> # Loop over each classifier and measure training time
>>> for name, classifier, param_grid  in zip ( names, estimators, param_grids) :
        print(f"Training {name}...")
        start_time = time.time()  # Start time

        # Define grid of parameters for GridSearchCV

        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X, y)

        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Training {name} took {elapsed_time:.2f} seconds\n")
Out[8]:
Training SVC...
Training SVC took 0.15 seconds

Training DecisionTreeClassifier...
Training DecisionTreeClassifier took 0.17 seconds

Training LogisticRegression...
Training LogisticRegression took 0.13 seconds
>>> 
>>> # (2)  Parallelize approach
>>> 
>>> parallelize_estimators(estimators, param_grids, X, y, optimizer ='GridSearchCV', 
                       pack_models = True )
Optimizing Estimators: 100%|###################| 3/3 [00:00<00:00, 33.22it/s] 
>>>          
```
When we check the time for three classifiers, the parallelized approach (`elapsed time =00s`) is 
much faster than the naive  approach (`elapsed time =0.45s`). Commonly, after performing 
several tests with a complex and large dataset on different computers with different processor units,
the ``gofast`` parallelized approach remains around ten times faster than the 
naive search.  

7. **Plot feature importances** 

``gofast`` provides some useful visualization utilities. For instance, the feature contributions 
with  the ``RandomForestClassifier`` can be displayed via the ``plot_rf_feature_importances``. 
Here is an example:

```python 
>>> from sklearn.ensemble import RandomForestClassifier 
>>> from gofast.plot import plot_rf_feature_importances 
>>> # Try to scale the numeric data 
>>> num_scaled = gf.soft_scaler (data_imputed[num_features],)  
>>> plot_rf_feature_importances (RandomForestClassifier(), num_scaled, yenc) 
``` 
There are several other tools provided by ``gofast`` that can effectively help users  to 
fast handling intricate dataset, manipulating features, training models, and perfoming 
anlyses and maths operations. 

## Note 

**gofast** is still under development and should be available for the community soon.
 
 
## Contributions 

_Join us in making machine learning workflows faster and more efficient. With gofast, you're not just processing data; you're accelerating toward breakthroughs and innovations._
