# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:53:55 2020

@author: Liping Wu
"""
'''
2-	Load the 'Iris.csv' file into a dataframe name the dataframe data_firstname_i where first name is your first name carry out the following activities:
a.	Display the column names
b.	Display the shape of the data frame i.e number of rows and number of columns
c.	Display the main statistics of the data
d.	Display the types of columns
e.	Display the first five records
f.	Find the unique values of the class
'''

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab08Wk10/"
filename = "iris.csv"
fullpath = os.path.join(path,filename)
print(fullpath)
data_liping_i = pd.read_csv(fullpath,sep=',')
print('Columns:',data_liping_i.columns.values)
print('Shape(rows and columns):',data_liping_i.shape)
print('Describe:\n',data_liping_i.describe())
print('DataType:\n',data_liping_i.dtypes) 
print('First 5 rows:\n', data_liping_i.head(5))
print('Unique Species:\n',data_liping_i['Species'].unique())

'''
3-	Separate the predictors from the target then split the dataset 
using numpy random function.
Following is the code, make sure you update the the data frame name correctly:
'''
# change columns to list
colnames=data_liping_i.columns.values.tolist()
print(colnames)
# collect the first four columns as predictors
predictors=colnames[:4]
print(predictors)
#assign the 5th collomn as target(or predict results)
target=colnames[4]
print(target)

import numpy as np
# add one column 'is_train' to the dataframe
data_liping_i['is_train'] = np.random.uniform(0, 1, len(data_liping_i)) <= .75
print(data_liping_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_liping_i[data_liping_i['is_train']==True], data_liping_i[data_liping_i['is_train']==False]
print('dataframe train:\n',train)
print('dataframe test:\n',test)
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

'''
4-	Build the decision tree using the training dataset. 
 Use enotrpy as a method for splitting, and split only when reaching  20 matches.
'''
from sklearn.tree import DecisionTreeClassifier
dt_liping = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_liping.fit(train[predictors], train[target])

'''
5-	Test the model using the testing dataset and calculate a confusion matrix this time using pandas
Following is the code, make sure you update model name correctly:
    '''
preds=dt_liping.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

'''
6-	Generate a dot file and visualize the tree using the online viz graph editor and share (download) as picture.
'''
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab08Wk10/"
from sklearn.tree import export_graphviz
with open(path+ 'dtree3.dot', 'w') as dotfile:
    export_graphviz(dt_liping, out_file = dotfile, feature_names = predictors)
dotfile.close()

'''
7-	Let us build the tree classifier again but this time let us split the data into 80% for training and 20% for testing:
''' 
X=data_liping_i[predictors]
Y=data_liping_i[target]

#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

'''
8-	 Let us now build the tree using the training as follows:
a.	Set the tree parameters
b.	Fit the training data
c.	Use the cross validation module and carry out a ten cross validation
d.	Use the sklearn metrics module to generate the score for the cross validation(i.e. build the model 10 times)
e.	Print the mean of the ten time runs
'''
depth = range(1,11)
for d in depth:    
    print("***********Max_depth=" + str(d) + "**************************")
    dt1_liping = DecisionTreeClassifier(criterion='entropy',max_depth=d, min_samples_split=20, random_state=99)
    dt1_liping.fit(trainX,trainY) 
    
    #check feature importance for each dataset
    importance = dt1_liping.feature_importances_

    # summarize feature importance
    for i,v in enumerate(importance):
    	print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    from matplotlib import pyplot
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.xlabel('Dataset (max_depth=' + str(d)+') features')
    pyplot.ylabel('importances')
    pyplot.show()
    
    # 10 fold cross validation using sklearn and all the data i.e validate the data 
    from sklearn.model_selection import KFold
    #help(KFold)
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    from sklearn.model_selection import cross_val_score
    score = np.mean(cross_val_score(dt1_liping, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    print('max_depth=' + str(d)+', score: ' , score)
    
    '''
    9-	Now let us test the model using the testing data i.e. the 20%:
    a.	Use the predict method and pass the 20% test data without labels i.e testX. 
    This should generate the predicted data store it in testY_predict
    b.	Use the metrics module from sklearn to calculate the score and the confusion matrix.
    '''
    ### Test the model using the testing data
    testY_predict = dt1_liping.predict(testX)
    testY_predict.dtype
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics 
    labels = Y.unique()
    print(labels)
    print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
    #Let us print the confusion matrix
    from sklearn.metrics import confusion_matrix
    print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))
    
    '''
    10-	Use Seaborn heatmaps to print the confusion matrix in a more clear and fancy way
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt     
    cm = confusion_matrix(testY, testY_predict, labels)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    
    # labels, title and ticks    
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix (max_depth=' + str(d)+')'); 
    ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']);
    plt.show()


'''
Exercises
1.	Prune the tree exercise
Change the max depth and re-run the model 10 times using some of the above code (you need to decide which code)
 each time with a different value of max depth ranging from 1 to 10 record your results on a table. 
 It would be nice if you could have a loop to automate. Show the results to your professor.
'''
# Please see code line 100-line 158

'''
2.	Do a feature importance test to determine which of the variables in the preceding dataset are actually
 important for the model. Share results with your professor.
'''
#ref:https://machinelearningmastery.com/calculate-feature-importance-with-python/
# decision tree for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

# define dataset
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab08Wk10/"
filename = "iris.csv"
fullpath = os.path.join(path,filename)
print(fullpath)
data_liping_i = pd.read_csv(fullpath,sep=',')
# change columns to list
colnames=data_liping_i.columns.values.tolist()
print(colnames)
# collect the first four columns as predictors
predictors=colnames[:4]
print(predictors)
#assign the 5th collomn as target(or predict results)
target=colnames[4]
print(target)

import numpy as np
# add one column 'is_train' to the dataframe
data_liping_i['is_train'] = np.random.uniform(0, 1, len(data_liping_i)) <= .75
print(data_liping_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_liping_i[data_liping_i['is_train']==True], data_liping_i[data_liping_i['is_train']==False]
'''
print('dataframe train:\n',train)
print('dataframe test:\n',test)
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
'''
# define the model
from sklearn.tree import DecisionTreeClassifier
dt_liping = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
# fit the model
dt_liping.fit(train[predictors], train[target])

# get importance
importance = dt_liping.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.xlabel('feature')
pyplot.ylabel('importances')
pyplot.show()


