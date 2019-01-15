# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:44:18 2018

@author: Mathieu
"""

from sklearn.base import ClassifierMixin
from sklearn.utils.testing import all_estimators
classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
print(classifiers)
names = [ i[1] for i in classifiers]