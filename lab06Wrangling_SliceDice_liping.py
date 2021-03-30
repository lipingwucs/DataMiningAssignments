# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:10:32 2020
@author: liping
#Sub setting the data slicing and dicing
"""

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'Customer Churn Model.txt'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping.columns.values

# extract one column (i.e. a series)
account_length=data_liping['Account Length']
account_length.head()
type(account_length)

#extract many columns into a new dataframe
subdata_liping = data_liping[['Account Length','VMail Message','Day Calls']]
subdata_liping.head()
type(subdata_liping)

# Create a list of wanted columns
wanted_columns=['Account Length','VMail Message','Day Calls']
subdata_liping=data_liping[wanted_columns]
subdata_liping.head()

## Another way useful when many columns
wanted=['Account Length','VMail Message','Day Calls']
column_list=data_liping.columns.values.tolist()
sublist=[x for x in column_list if x not in wanted]
subdata=data_liping[sublist]
subdata_liping.head()

## Rows
#Select the first 50 rows
data_liping[:50]


# select 50 rows starting at 25
data_liping[25:75]

# filter the rows that have clocked day Mins to be greater than 350. 
sub_data_liping=data_liping[data_liping['Day Mins']>350]
sub_data_liping.shape
sub_data_liping

#filter the rows for which the state is VA:
sub_data_liping=data_liping[data_liping['State']=='VA']
sub_data_liping.shape
sub_data_liping


#filter the rows that have clocked day Mins to be greater than 250 and the state value is VA
sub_data_liping=data_liping[(data_liping['Day Mins']>250)&(data_liping['State']=='VA')]
sub_data_liping.shape
sub_data_liping[['State','Day Mins']]

## Create a new column for total minutes
data_liping['Total Mins']=data_liping['Day Mins']+data_liping['Eve Mins']+data_liping['Night Mins']
data_liping['Total Mins'].head()





