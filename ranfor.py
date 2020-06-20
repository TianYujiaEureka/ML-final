# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:16:30 2020

@author: Eureka
"""


import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
os.chdir('D:\\ice_data\\train_clean')
ice_data_15=pd.read_csv('train_cleandata_15.csv',sep=',')
ice_data_21=pd.read_csv('train_cleandata_21.csv',sep=',')
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import copy
from sklearn.preprocessing import MinMaxScaler
label_15=ice_data_15.label
ice_data_15 = ice_data_15.drop('label',axis=1)

X_train, X_test1, y_train, y_test1 = train_test_split(ice_data_15, label_15, test_size=0.1, random_state=128, shuffle = True)# shuffle默认为True
# 在选择的数据中，选择2/3作为训练集，1/3作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.7, random_state=128, shuffle = False)# shuffle默认为True

# 归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)
# 使用随机森林分类器（直接使用网格搜索的最佳参数）
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10 ,max_depth=10,random_state=24, n_jobs=-1)
rf_clf.fit(X_train, y_train)
y_pre = rf_clf.predict(X_test)
#print(rf_clf.oob_score_)
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_pre))
print('泛化tree')
label_21=ice_data_21.label
ice_data_21 = ice_data_21.drop('label',axis=1)
X_test_scaled_21 = min_max_scaler.fit_transform(ice_data_21)
y_pre_21=rf_clf.predict(X_test_scaled_21)
print(classification_report(y_true=label_21, y_pred=y_pre_21))