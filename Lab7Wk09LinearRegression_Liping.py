# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:20:03 2020

@author: LipingWu
"""

#2-	Load the 'Adertising.csv' file into a dataframe 
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab07Wk09/"
filename = 'Advertising.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
data_liping_adv = pd.read_csv(fullpath)
data_liping_adv.columns.values
data_liping_adv.shape
data_liping_adv.describe()
data_liping_adv.dtypes
data_liping_adv.head(5)


'''3-	Let us check if there is a correlation between advertisement costs on TV and the resultant sales. Remember the formula:
 
a.	Use the numpy package to build a function to calculate the correlation between each input variable TV,Radio & Newspaper and the output Sales
b.	Run the below code snippet , you should get a result the following results:
0.782224424861606
0.5762225745710553
0.22829902637616525'''

import numpy as np
# method corrcoeff
def corrcoeff(df,var1,var2):
    df['corrn']=(df[var1]-np.mean(df[var1]))*(df[var2]-np.mean(df[var2]))
    df['corrd1']=(df[var1]-np.mean(df[var1]))**2
    df['corrd2']=(df[var2]-np.mean(df[var2]))**2
    corrcoeffn=df.sum()['corrn']
    corrcoeffd1=df.sum()['corrd1']
    corrcoeffd2=df.sum()['corrd2']
    corrcoeffd=np.sqrt(corrcoeffd1*corrcoeffd2)
    corrcoeff=corrcoeffn/corrcoeffd
    return corrcoeff
print(corrcoeff(data_liping_adv,'TV','Sales'))
print(corrcoeff(data_liping_adv,'Radio','Sales'))
print(corrcoeff(data_liping_adv,'Newspaper','Sales'))

# 4 Use  the matplotlib module to visualize the  relationships between each of the inputs and the output (sales),
# i.e. generate three scattered plots.
import matplotlib.pyplot as plt
plt.plot(data_liping_adv['TV'], data_liping_adv['Sales'],'ro', color='blue')
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title('TV vs Sales')
plt.show()

plt.plot(data_liping_adv['Radio'], data_liping_adv['Sales'],'ro', color='green')
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.title('Radio vs Sales')

plt.plot(data_liping_adv['Newspaper'], data_liping_adv['Sales'],'ro', color ='red')
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.title('Newspaper vs Sales')

'''	Use the ols method and the statsmodel.formula.api
 library to build a linear regression model with TV costs as the predictor (input) and sales as the predicted
 i.e. estimate the parameters of the model. You should get the following results:'''
import statsmodels.formula.api as smf
model1=smf.ols(formula='Sales~TV',data=data_liping_adv).fit()
model1.params	

'''
model2=smf.ols(formula='Sales~Radio',data=data_liping_adv).fit()
model2.params	

model3=smf.ols(formula='Sales~Newspaper',data=data_liping_adv).fit()
model3.params	
'''

#5-	Generate the p-values and the R-squared and model summary, run the following lines of code
print(model1.pvalues)
print(model1.rsquared)
print(model1.summary())

'''
print(model2.pvalues)
print(model2.rsquared)
print(model2.summary())

print(model3.pvalues)
print(model3.rsquared)
print(model3.summary())
'''

'''
6-	Re-build the model with two predictors TV and Radio as input variables and print the parameters, p-values, rsquared and summary. Then:	
a.	Create a new data frame with 2 new values for TV and Radio
b.	Predict using the new values
c.	Change the values and run the prediction again
d.	Change the values again to two values already existing in the dataset and run the prediction again
7-	Based on the output our new formula is:  
'''

import statsmodels.formula.api as smf
model3=smf.ols(formula='Sales~TV+ Radio',data= data_liping_adv).fit()
print(model3.params)
print(model3.rsquared)
print(model3.summary())

## Predicte a new value
X_new2 = pd.DataFrame({'TV': [50], 'Radio' : [40] })
# predict for a new observation
sales_pred2=model3.predict(X_new2)
print(sales_pred2)

'''
8-	In this step we will build the model using scikit-learn package, this is the more commonly used package to build data science projects. This method is more elegant as it has more in-built methods to perform the regular processes associated with regression. Carry out the following:
a.	Import the necessary modules
b.	Split the dataset into 80% for training and 20% for testing
c.	Print out the parameters 
d.	Test the model using the Train/Test
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

feature_cols = ['TV', 'Radio']
# feature_cols = ['TV', 'Radio', 'Newspaper'] if more column invols
X = data_liping_adv[feature_cols]
Y = data_liping_adv['Sales']
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.20)
trainX.describe()
trainX.columns.values
trainX.shape

trainY.describe()
trainY.shape

testX.describe()
testX.columns.values
testX.shape

testY.describe()
testY.shape

#print test values
for z in zip(testX.index, testY):
    print (z)


# to get Linear Regression
lm = LinearRegression()
lm.fit(trainX, trainY)
print (lm.intercept_)
print (lm.coef_)
for z in zip(feature_cols, lm.coef_):
    print (z)
   
# to predict TestY base on the Linear Regression
lm.score(trainX, trainY)
predictY = lm.predict(testX)
for p in zip(testX.index, testY, predictY):
    print (p)


'''9-	Feature selection: using the scikit , in order to check which predictors
 are best as input variable to the model run the following code sinpet '''
 
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
feature_cols = ['TV', 'Radio','Newspaper']
X = data_liping_adv[feature_cols]
Y = data_liping_adv['Sales']
estimator = SVR(kernel="linear")
selector = RFE(estimator,2,step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.ranking_)
for s in zip(selector.support_,selector.ranking_ ):
    print(s)
    






