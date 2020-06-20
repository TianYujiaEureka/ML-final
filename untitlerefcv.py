# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:04:03 2020

@author: Eureka
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import copy
os.chdir('D:\\ice_data\\train_clean')
ice_data_15=pd.read_csv('train_cleandata_15.csv',sep=',')
ice_data_21=pd.read_csv('train_cleandata_21.csv',sep=',')
label_15=ice_data_15.label
ice_data_15 = ice_data_15.drop('label',axis=1)
ice_data_15.head()

#svc=SVC(kernel="linear")
dt = DecisionTreeClassifier()
X_train=copy.deepcopy(ice_data_15)
y_train=label_15
rfecv = RFECV(estimator=dt, step=2, cv=StratifiedKFold(3), scoring='roc_auc')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features names: %s" % X_train.columns[rfecv.support_])
print("Ranking of features nums: %s" % rfecv.ranking_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig("feature.png")
plt.show()
