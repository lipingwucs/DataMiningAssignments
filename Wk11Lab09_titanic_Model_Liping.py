# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:44:51 2020

@author: Liping Wu
"""

import pandas as pd
import os
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width

# Load the dataset in a dataframe object 
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab09Wk11/"
filename = 'titanic3.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
df = pd.read_csv(fullpath)


# Explore the data check the column values
print(df.columns.values)
print (df.head())
print (df.info())
categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
          
print(categories)
type(categories)        #list
print(df.columns.values)
print(df.head())
df.describe()
df.info()
#check for null values
print(len(df) - df.count())  #Cabin , boat, home.dest have so many missing values

'''
7.	Carry out a detailed exploration and modeling for the below features
o	Save the features into a dataframe named df_firstname where firstname is your firstname.
o	Print the first five records
o	Print the column values
o	Print the unique values for all the three features sex, embarked and age.
o	Print the unique values for the class survived
o	Check the null values
'''
include = ['age','sex', 'embarked', 'survived']
df_ = df[include]
print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['sex'].unique() #array(['female', 'male', nan], dtype=object)
df_['embarked'].unique() # array(['S', 'C', nan, 'Q'], dtype=object)
df_['age'].unique()
df_['survived'].unique()
# check the null values
print(df_.isnull().sum())
print(df_['sex'].isnull().sum()) #1
print(df_['embarked'].isnull().sum()) #3
print(len(df_) - df_.count())

# 8.	Drop the rows with missing values
df_.loc[:,('age','sex', 'embarked', 'survived')].dropna(axis=0,how='any',inplace=True) # ： means for all rows of columns()
df_.info() 

'''
9.	Identify the features with categorical values
'''
df_.dtypes
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
          print(categoricals)
     
print(categoricals) #['sex', 'embarked']
'''
10.	Convert the categorical values into numeric columns using the get dummies
o	Create a new dataframe named df_ohe_fristname where firstname is your firstname.(notice this is a different way of calling get¬_dummies)
o	Set the column display
o	Display the first five records
o	Check the column values
o	Check for any missing values
'''
df_.info()   #[ age     sex  embarked   survived]
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values) # ['age' 'survived' 'sex_female' 'sex_male' 'embarked_C' 'embarked_Q'  'embarked_S']
#check for null values
print(len(df_ohe) - df_ohe.count())

'''
You should have all numeric data with no missing values 1307 rows and 7 columns 
11.	Standardize the data i.e. mean of zero
o	Import preprocessing library from sklearn
o	Save the column name into a variable called names
o	Create a scalar object
o	Create a new data frame name it scaled_df_firstname where firstname is your firstname, fit the data into the scalar object. Note that this object is an array, therefore you will need to reconvert it into a dataframe  using pandas (pd.DataFrame) and reattach the column names you saved
o	Check the new numeric values (mean, standard deviation, min, max , median)
'''
from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
print(names)  # Index(['age', 'survived', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S'], dtype='object')
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['age'].describe())
print(scaled_df['sex_male'].describe())
print(scaled_df['sex_female'].describe())
print(scaled_df['embarked_C'].describe())
print(scaled_df['embarked_Q'].describe())
print(scaled_df['embarked_S'].describe())
print(scaled_df['survived'].describe())
print(scaled_df.dtypes)
print(scaled_df.columns)
# # Index(['age', 'survived', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S'], dtype='object')
'''
12.	Build the logistic regression model 
o	Import the model
o	Define the dependent variable
o	Select the feature using the difference method 
o	Convert the class from float back to integer using the  astype method
o	Split the data into 80% for taining and 20% for testing
o	Validate score the model using 10 fold cross validation
o	Print the final score
Below is the code make sure to rename the dataframe
'''

from sklearn.linear_model import LogisticRegression
dependent_variable = 'survived'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]   #select the columns which is not survived as x 
print(x.columns)
 # ['age', 'embarked_C', 'embarked_Q', 'embarked_S', 'sex_female', 'sex_male']
x.dtypes

##first check null values
x.isna().sum()  # check null value：  age   264
#fillna
x.fillna(0, inplace=True)
x.isna().sum()  # check null value： age   0


y = scaled_df[dependent_variable]
y.dtypes   # dtype('float64')
type(y)   # pandas.core.series.Series
#convert the class back into integer
##first check null values
y.isna().sum()  # check null value： 1
#fillna
y.fillna(0, inplace=True)
y.isna().sum()  # check null value： 0
y = y.astype(int)
y.dtypes

# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)



'''
13.	Test the model using the 20% testing data
o	Use the predict method
o	Import the metrics module
o	Print the accuracy
o	Print the confusion matrix
'''
testY_predict = lr.predict(testX)
testY_predict.dtype

#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))  # compare the real value and predict rate
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

'''
14.	Serialize (save) the model as an object
o	Import joblib
o	Use the dump method to create the model pickle object
'''
import joblib 
joblib.dump(lr, 'D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab09Wk11/model_lr2.pkl')
print("Model dumped!")

'''
15.	Serialize save the model columns as an object
'''
model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, 'D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab09Wk11/model_columns.pkl')
print("Models columns dumped!")

