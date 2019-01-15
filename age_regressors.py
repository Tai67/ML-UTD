# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:31:32 2019

@author: Mathieu
"""
import copy
import xgboost
import numpy as np
import modelslists
import tpot
import pandas as pd
from transformers import nichts

from scorer import roc_auc_scorer
from numpy.random import seed

from scorer import kappa
from ds_filters import ds_2CAT
from preprocessing import data_preprocessing

import sklearn
from sklearn import naive_bayes, linear_model
from sklearn import tree, svm, preprocessing
from sklearn import ensemble, neighbors, discriminant_analysis

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score, make_scorer

from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler


model1 = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LinearSVR(C=15.0, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.0001)),
    SelectFwe(score_func=f_regression, alpha=0.034),
    ExtraTreesRegressor(bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=20, min_samples_split=13, n_estimators=100)
)

model2 = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.30000000000000004, tol=0.01)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=8, min_samples_split=14, n_estimators=100)
)

name='Age'
classification=True
scorer=kappa
greater_is_better=True
gender = None    
s_filter=ds_2CAT
drop=['SCOREBDI']
target='ds'
data_choice = 'cv'
config_dict=None
pop = 250
gen = 20
th = 19
save=True
save_d=True
c_display=0
norm=False
clean=False
dynamic_drop = False
factor = 3
ban_thresh = 10
undersample = None
age =  None
rand = 23        

df = data_preprocessing(name=name, s_filter=s_filter, 
                            drop=drop, data_choice = data_choice, 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh, gender=gender, age=age, 
                            undersample =undersample, rand=rand)