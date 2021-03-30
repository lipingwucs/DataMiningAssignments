# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:22:00 2020

@author: liping wu
"""
# Load data
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab07Wk09/"
filename = 'Bank.csv'
fullpath = os.path.join(path,filename)
data_liping_b = pd.read_csv(fullpath,sep=';')
print(data_liping_b.columns.values)
print(data_liping_b.shape)
print(data_liping_b.describe())
print(data_liping_b.dtypes) 
print(data_liping_b.head(5))

# 1-change the y column from object to integer
print(data_liping_b)
data_liping_b['y']=(data_liping_b['y']=='yes').astype(int)
print(data_liping_b)



# Cleanup data
# Before cleaning
print(data_liping_b['education'].unique())
import numpy as np
# change 'basic.9y'to 'Basic'
data_liping_b['education']=np.where(data_liping_b['education'] =='basic.9y', 'Basic', data_liping_b['education'])
print(data_liping_b['education'].unique())

# change 'basic.6y'to 'Basic'
data_liping_b['education']=np.where(data_liping_b['education'] =='basic.6y', 'Basic', data_liping_b['education'])
print(data_liping_b['education'].unique())

# change 'basic.4y'to 'Basic'
data_liping_b['education']=np.where(data_liping_b['education'] =='basic.4y', 'Basic', data_liping_b['education'])
print(data_liping_b['education'].unique())

# change'professional.course' to 'Professional Course'
data_liping_b['education']=np.where(data_liping_b['education'] =='university.degree', 'University Degree', data_liping_b['education'])

# change 'basic.4y'to 'Basic'
data_liping_b['education']=np.where(data_liping_b['education'] =='professional.course', 'Professional Course', data_liping_b['education'])

# change 'high.school' to 'High School',
data_liping_b['education']=np.where(data_liping_b['education'] =='high.school', 'High School', data_liping_b['education'])
data_liping_b['education']=np.where(data_liping_b['education'] =='illiterate', 'Illiterate', data_liping_b['education'])
data_liping_b['education']=np.where(data_liping_b['education'] =='unknown', 'Unknown', data_liping_b['education'])
# After cleaning
print(data_liping_b['education'].unique())


#Check the values of who  purchased the deposit account
print(data_liping_b['y'].value_counts())


#Check the average of all the numeric columns
pd.set_option('display.max_columns',100)
print(data_liping_b.groupby('y').mean())

#Check the mean of all numeric columns grouped by education
print(data_liping_b.groupby('education').mean())

#Check the mean of all numeric columns grouped by marital
print(data_liping_b.groupby('marital').mean())


#Plot a histogram showing purchase by education category
import matplotlib.pyplot as plt
pd.crosstab(data_liping_b.education,data_liping_b.y)
pd.crosstab(data_liping_b.education,data_liping_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Level')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')


#draw a stacked bar chart of the marital status and the purchase of term deposit to see whether this can be a good predictor of the outcome
table=pd.crosstab(data_liping_b.marital,data_liping_b.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')


#plot the bar chart for the Frequency of Purchase against each day of the week to see whether this can be a good predictor of the outcome
pd.crosstab(data_liping_b.day_of_week,data_liping_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')


#Repeat for the month
pd.crosstab(data_liping_b.month,data_liping_b.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')

#Plot a histogram of the age distribution
data_liping_b.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

#Plot a histogram of the age distribution
data_liping_b.month.hist()
plt.title('Histogram of month')
plt.xlabel('Month')
plt.ylabel('Frequency')

#Plot a histogram of the education distribution
data_liping_b.education.hist()
plt.title('Histogram of Education')
plt.xlabel('Education')
plt.ylabel('Frequency')


#4 Deal with the categorical variables, use a for loop
#1- Create the dummy variables 
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(data_liping_b[var], prefix=var)    
    print(cat_list)   
cat_list.describe
    
print(data_liping_b)

#data_liping_b1=data_liping_b.join(cat_list,how= 'inner')
data_liping_b = pd.merge(data_liping_b, cat_list,left_index=True, right_index=True, how='outer')
# to observe the datachange
data_liping_b.columns.values
data_liping_b.describe()
data_liping_b.head()
data_liping_b.describe

   
#  2- Removee the original columns
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_liping_b_vars=data_liping_b.columns.values.tolist()
to_keep=[i for i in data_liping_b_vars if i not in cat_vars]
data_liping_b_final=data_liping_b[to_keep]
data_liping_b_final.columns.values
data_liping_b_final.describe
    
# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
data_liping_b_final_vars=data_liping_b_final.columns.values.tolist()
Y=['y']
X=[i for i in data_liping_b_final_vars if i not in Y ]
type(Y)
type(X)

##5-	Carryout feature selection and update the data
#1- We have many features so let us carryout feature selection

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 12)
rfe = rfe.fit(data_liping_b_final[X],data_liping_b_final[Y] )
print(rfe.support_)
print(rfe.ranking_)
for r in zip(rfe.support_, rfe.ranking_ ):
    print(r)

#2- Update X and Y with selected features
cols=['previous', 'euribor3m',  'poutcome_success', 'poutcome_failure'] 
data_liping_b_final.columns.values
X=data_liping_b_final[cols]
Y=data_liping_b_final['y']
type(Y)
type(X)

'''6-	Build the logistic regression model as follows:
a.	Split the data into 70%training and 30% for testing
b.	Build the model using “sklearn  linear_model.LogisticRegression” 
c.	Fit the training data
d.	Validate the parameters and check model accuracy'''

#1- split the data into 70%training and 30% for testing, note  added the solver to avoid warnings
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train.describe
Y_train.describe
X_test.describe
Y_test.describe 


# 2-Let us build the model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(X_train, Y_train)


#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
for p in zip(probs, predicted ):
    print([p])
#4-Check model accuracy
print (metrics.accuracy_score(Y_test, predicted))

'''7-	To avoid sampling bias run cross validation for 10 times, as follows:
a.	Use the cross_val_score module from sklearn.model_selection and set the parameters
b.	Save the results of each run in scores
c.	Produce the mean
Following is the code:'''

from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X, Y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

'''8-	Generate the confusion matrix as follows:
a.	Prepare two arrays one for the predicted values Y_P and one for actual values Y_A of the test. For the predicted use a threshold of 0.05, this means if the probability is higher than 0.05 the model will classify the instance as 1 and if it is lower than 0.05 it will be classified as 0.
b.	Use the confusion_matrix option from the sklearn.metrics module to generate the matrix
Following is the code:'''
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
import numpy as np
Y_A =Y_test.values
Y_P = np.array(prob_df['predict'])
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_A, Y_P)
print (confusion_matrix)




