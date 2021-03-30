# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:33:59 2020
@author: liping
"""
####Imputation 
# Fill the missing values with zeros
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)

data_liping.fillna(0,inplace=True)
data_liping.head()


import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping.fillna("missing",inplace=True)
data_liping.head(30)

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping['body'].head(10)

data_liping['body'].fillna("missing",inplace=True)
data_liping['body'].head(30)

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping['age'].head(10)
## get the age mean
ave_age= data_liping['age'].mean()
print('Average age of 10 is: ', ave_age)

data_liping['age'].fillna(data_liping['age'].mean(),inplace=True)
data_liping['age'].head(30)



# Creating Dummy
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping.columns.values

dummy_sex=pd.get_dummies(data_liping['sex'],prefix='sex')
dummy_sex.head()

column_name=data_liping.columns.values.tolist()
column_name
column_name.remove('sex')
column_name
data_liping[column_name].join(dummy_sex)






