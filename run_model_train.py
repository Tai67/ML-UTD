# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:59:52 2018

@author: Mathieu
"""

import numpy as np
import pickle
import pandas as pd
import pytest
import transformers

from preprocessing import data_preprocessing
from scorer import f1_scorer
from scorer import kappa
from scorer import roc_auc_scorer

import time

from ds_filters import ds_2CAT
from ds_filters import ds_2CAT_b
from ds_filters import ds_4CAT
from transformers import nichts

from sklearn.base import clone

from experiment_functions import assign_filename
from experiment_functions import pipeline_eval
from experiment_functions import pipeline_maker
from experiment_functions import make_an_excel

from display_functions import pipeline_results_display
from display_functions import pipeline_retest_results_display

from numpy.random import seed
from sklearn.model_selection import cross_val_score

from sklearn.utils.testing import all_estimators
from sklearn.base import ClassifierMixin

from sklearn.metrics import accuracy_score, log_loss, balanced_accuracy_score,auc
from sklearn.metrics import f1_score, cohen_kappa_score, make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#---------------------------
    
def all_classifier_search(file_name='test', s_filter=ds_2CAT_b, scoring = None, 
                 drop=['SCOREBDI'], data_choice = 'cv', 
                 pop = 500, gen = 10, th = 19,save=True, display=True, 
                 classification=True, greater_is_better=True,config_dict=None,
                 c_display=1, gender = None, age = None,
                 save_d= True,
                 target='ds',
                 norm= False,
                 clean=False,
                 factor = 3, dynamic_drop=False, ban_thresh = 10, rand = 23, 
                 cv = 3, undersample = None, dummy= False, 
                 func_drop_list=[[nichts,[]]], model = False):
    """
    1. Takes dataset configuration
    2. Saves experiment variables
    3. Builds the scorer from metric functions
    4. Takes a list of all classifiers available in scikit
    5. Test them with cross-validation
    6. Returns a dict composed from 'classifier':result_dict
    """
    f_names = []
    for func_d in func_drop_list:
        f_names.append(func_d[0].__name__)
    z = locals()
    pd.Series(z).to_csv(path=(file_name+' experiment variables.csv'))

    if scoring == None :
        
                scoring_c={'log_loss': log_loss, 
                 'accuracy':accuracy_score,
            'balanced_accuracy_score':balanced_accuracy_score,
            'precision':    lambda x,y :precision_score(x,y, average='macro'),
            'recall':       lambda x,y :recall_score(x,y, average='macro'),
            'auc':roc_auc_scorer}
    else : scoring_c=scoring.copy()
    seed(rand)
#    print(file_name)
#    with open(file_name+'locals.pickle', 'wb') as handle:
#        pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(str(scoring_c['accuracy']))
    
    for key in scoring_c.keys():
        x = make_scorer(scoring_c[key])
        scoring_c[key]= x
    
    df = data_preprocessing(name=file_name, s_filter=s_filter, 
                        drop=drop, data_choice = data_choice, 
                        normalization=norm ,clean= clean, th=th,
                        factor = factor, dynamic_drop=dynamic_drop,
                        ban_thresh = ban_thresh, gender = gender, age =age,
                        undersample = undersample, func_drop_list=func_drop_list)
    df.to_csv(path_or_buf=(file_name+' dataset.csv'))
    classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
    results=dict()
    for name, clas in classifiers:
#        print(name,':',clas)
        try :
            scores = cross_validate(clas(),  df.drop(target, axis =1),\
                                    df[target], cv=cv, scoring = scoring_c)
            results[name]=scores
        except TypeError :
            print(name,' doesn\'t have fit method')
        except Exception as e: print(e)
    return results


def check_all_classifiers(datachoice='cv', rand=23, undersample=1, save=False, 
                          threshold=19,scoring={'accuracy':accuracy_score,
                                   'cohen kappa score':kappa,
#                                   'f1 score':f1_score,
                                   'balanced accuracy':balanced_accuracy_score,
                                   'log loss':log_loss} , trans_f=[[nichts,[]]],
                                                age = None, file_name='test', drop=['SCOREBDI']):
    """
    1. Throw all parameters in all_classifier_search, gets the class/res in result
    2. Averages the values
    """
    
    r = all_classifier_search(rand=rand, scoring = scoring, th=threshold, 
                              data_choice=datachoice, func_drop_list=trans_f,
                                    undersample=undersample,age=age, 
                                    file_name=file_name, drop=drop)
    
    this, end_keys = dict(),list()
    for model in r.keys():
        this[model]=dict()
        for values in r[model].keys():
            this[model][values]=np.mean(r[model][values])
            
    result = pd.DataFrame.from_dict(this).transpose()
    result = result.sort_values(by='test_balanced_accuracy',ascending=False)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for key in result.keys():
        if 'test' in key:
            end_keys.append(key)
    result = result[end_keys]
    if save : result.to_csv((file_name+'/Results of traditionnal ML - '+datachoice+\
                             ' rand -'+str(rand)+' us -'+str(undersample)+str(timestr)+'.csv'))
    print('-------------\n')
    print('-------------')
    return result.index

#    l = ['cv', 'ct', 'both', 'rl_cv', 'rl_ct', 'rl_both']
#    l = ['cv','rl_cv']
#              [[transformers.add_total, []], [transformers.normalize, ['sum', 'Age', 'Gender','ds','SCOREBDI']]],
#              [[transformers.add_total, []], [(lambda x,y : transformers.normalize(x,y,mult=True)), \
#                ['sum', 'Age', 'Gender','ds','SCOREBDI']]],

def main(rand=27, undersample = None, threshold=19, 
         trans_functions_o= [[[nichts,[]]]], age=None,data_choices=['cv'], name='test', 
         save=True, drop=['SCOREBDI']):    

    l = data_choices.copy()
    if trans_functions_o==None:
        trans_functions = [
              [[transformers.add_total, []], [transformers.drop_all, ['sum', 'Age', 'Gender','ds','SCOREBDI']]],
              [[transformers.add_total, []], [(lambda x,y : transformers.normalize(x,y,mult=True)), \
                ['sum', 'Age', 'Gender','ds','SCOREBDI']]],

              [[transformers.add_total, []], [(lambda x,y : transformers.normalize(x,y,mult=True,norm='l2')), \
                ['sum', 'Age', 'Gender','ds','SCOREBDI']]]
            ]
    else : trans_functions = trans_functions_o.copy()
              
    best_class=dict()
    for trans_function in trans_functions :
        for targ in l:
            seed(rand)
            file_name = assign_filename (str(str(targ)), save)
            best_class[targ] = (list(check_all_classifiers(datachoice=targ,
                  rand=rand, undersample=undersample,threshold=threshold, 
                  save=True,trans_f=trans_function,age=age, file_name=file_name,
                  drop=drop)))[:10]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            pd.DataFrame.from_dict(best_class).transpose().to_csv((file_name+'/Rank'+str(rand)+\
                        ' us '+str(undersample)+str(timestr)+'.csv'))
    return best_class

def cat4_conf():
    scoring = {'accuracy':accuracy_score,
           'precision':lambda x:precision_score(x, average='macro'),
           'recall':lambda x:recall_score(x, average='macro')}
    check_all_classifiers(datachoice='cv', rand=23, undersample=None, save=True, 
                      scoring=scoring,
                      threshold=[10, 19, 29])
    
    

