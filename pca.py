# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:02:44 2020

@author: Eureka
"""

from sklearn.decomposition import PCA  #PCA

'''
['wind_direction', 'pitch3_speed', 'wind_direction_mean',
       'pitch2_ng5_DC', 'acc_x', 'yaw_speed', 'int_tmp', 'pitch1_speed',
       'pitch2_angle', 'yaw_position', 'pitch1_ng5_DC', 'pitch3_ng5_DC',
       'pitch3_ng5_tmp', 'pitch2_speed', 'power', 'pitch1_angle',
       'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'generator_speed', 'pitch3_angle',
       'pitch1_moto_tmp', 'acc_y', 'pitch2_moto_tmp', 'pitch3_moto_tmp',
       'environment_tmp', 'group', 'wind_speed']'''
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
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
pca = PCA(n_components=8)
pca.fit(ice_data_15)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_)
print (pca.n_components_)
X_p =pca.transform(ice_data_15)
X_train, X_test1, y_train, y_test1 = train_test_split(X_p, label_15, test_size=0.1, random_state=128, shuffle = True)# shuffle默认为True
# 在选择的数据中，选择2/3作为训练集，1/3作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=128, shuffle = False)# shuffle默认为True

# 归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)
from sklearn.tree import DecisionTreeClassifier
y_pre=[]
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
y_pre=dt.predict(X_test_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_pre))
print('泛化pca')
label_21=ice_data_21.label
ice_data_21 = ice_data_21.drop('label',axis=1)
ice_data_21=pca.transform(ice_data_21)
X_test_scaled_21 = min_max_scaler.fit_transform(ice_data_21)
y_pre_21=dt.predict(X_test_scaled_21)
print(classification_report(y_true=label_21, y_pred=y_pre_21))