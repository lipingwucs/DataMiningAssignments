# -*- coding: utf-8 -*-

#Created on Tue Nov  3 12:36:16 2020
#author: liping

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
print("****************data load successfully*******************")


print("****************get first five records*******************")
data_liping.head()

print("****************get data shapes*******************")
data_liping.shape


print("****************get data columns values - method1 *******************")
data_liping.columns.values

print("****************get data columns values - method2 *******************")
print(data_liping.columns.values)

print("****************get data columns values - method3 *******************")
for col in data_liping.columns:
    print(col)
    
print("****************create summaries of data *******************")
data_liping.describe()

print("****************get the types of columns*******************")
data_liping.dtypes
