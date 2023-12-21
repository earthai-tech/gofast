# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:47:57 2023

@author: Daniel
"""


from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from gofast.models.optimize import parallelize_estimators 

X, y = load_iris(return_X_y=True)

def test_parallelize_estimators ( optimizer = 'RSCV', pack_models =False ): 
    estimators = [SVC(), DecisionTreeClassifier()]
    param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}
                       ]
    o=parallelize_estimators(estimators, param_grids, X, y, optimizer =optimizer, 
                           pack_models = pack_models )
    return o
    

if __name__=='__main__': 
    
    o= test_parallelize_estimators(optimizer ='GSCV', pack_models= True)
    