# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:05:08 2020

@author: liping wu 
"""
'''
2-	Load the 'wine.csv' file into a dataframe name the dataframe data_firstname_wine where first name is your first name carry out the following activities:
a.	Display the column names
b.	Display the shape of the data frame i.e number of rows and number of columns
c.	Display the main statistics of the data
d.	Display the types of columns
e.	Display the first five records
f.	Find the unique values of the quality attribute
g.	Find the mean of the various chemical compositions across samples for the different groups of the wine quality
'''
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab08Wk10/"
filename = 'wine.csv'
fullpath = os.path.join(path,filename)
data_liping_wine = pd.read_csv(fullpath,sep=';')

# set the columns display and check the data
pd.set_option('display.max_columns',15)
print(data_liping_wine.head())
print(data_liping_wine.columns.values)
print(data_liping_wine.shape)
print(data_liping_wine.describe())
print(data_liping_wine.dtypes) 
print(data_liping_wine.head(5))
print(data_liping_wine['quality'].unique())
pd.set_option('display.max_columns',15)
print(data_liping_wine.groupby('quality').mean())

'''
3-	Plot a histogram to see the number of wine samples in each quality type
Following is the code, make sure you update the the data frame name correctly:
'''
import matplotlib.pyplot as plt

plt.hist(data_liping_wine['quality'],bins=10)
plt.xlabel('quality')
plt.ylabel('sample amount')
plt.title('sample and quality')

'''
4-	Use seaborn library to generate different plots: histograms, pairplots, heatmaps…etc. and investigate the correlations.
Following are the code snippets, make sure you update the data frame name correctly:
    '''
import seaborn as sns
sns.distplot(data_liping_wine['quality'])

# plot only the density function
sns.distplot(data_liping_wine['quality'], rug=True, hist=False, color = 'r')
# Change the direction of the plot
sns.distplot(data_liping_wine['quality'], rug=True, hist=False, vertical = True)


# Check all correlations
sns.pairplot(data_liping_wine)
# Subset three column
x=data_liping_wine[['fixed acidity','chlorides','pH']]
y=data_liping_wine[['chlorides','pH']]
# check the correlations 
sns.pairplot(x)


# Generate heatmaps
sns.heatmap(data_liping_wine[['fixed acidity']])
sns.heatmap(x)
sns.heatmap(x.corr())
sns.heatmap(x.corr(),annot=True)
##
import matplotlib.pyplot as plt
plt.figure(figsize=(10,9))
sns.heatmap(x.corr(),annot=True, cmap='coolwarm',linewidth=0.5)


##line two variables
plt.figure(figsize=(20,9))
sns.lineplot(data=y) 
sns.lineplot(data=y,x='chlorides',y='pH')
## line three variables
sns.lineplot(data=x)

'''
5-	Normalize the data in order to apply clustering, the formula is as follows:
    Zi=(Xi-Xmin)/(Xmax-Xmin) 
Following is the code, make sure you update model name correctly:
    '''
data_liping_wine_norm = (data_liping_wine - data_liping_wine.min()) / (data_liping_wine.max() - data_liping_wine.min())
data_liping_wine_norm.head()
data_liping_wine_norm.shape

'''
6-	Generate some additional plots for the normalized data:
Following is the code, make sure you update model name correctly:
    '''
# check some plots after normalizing the data
x1=data_liping_wine_norm[['fixed acidity','chlorides','pH']]
y1=data_liping_wine_norm[['chlorides','pH']]
sns.lineplot(data=y1) 
sns.lineplot(data=x1)
sns.lineplot(data=y,x='chlorides',y='pH')
             
'''
7-	Cluster the data (observations) into 6 clusters using k-means clustering algorithm. 
8-	Following is the code, make sure you update model name correctly:
    '''
from sklearn.cluster import KMeans
from sklearn import datasets
n_clusters=3
# n_clusters=6
model=KMeans(n_clusters)
model.fit(data_liping_wine_norm)

'''
9-	Check the results as follows:
a.	Print the model labels
b.	Append the clusters to each record on the dataframe, i.e. add a new column for clusters
c.	find the final cluster's centroids for each cluster
d.	Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster. For an efficient cluster, the J-score should be as low as possible.
e.	plot a histogram for the clusters variable to get an idea of the number of observations in each cluster.
Following is the code, make sure you update model name correctly:
'''

model.labels_
# Append the clusters to each record on the dataframe, i.e. add a new column for clusters
md=pd.Series(model.labels_)
data_liping_wine_norm['clust']=md
data_liping_wine_norm.head(10)
#find the final cluster's centroids for each cluster
model.cluster_centers_
#Calculate the J-scores The J-score can be thought of as the sum of the squared distance between points and cluster centroid for each point and cluster.
#For an efficient cluster, the J-score should be as low as possible.
model.inertia_
#let us plot a histogram for the clusters
import matplotlib.pyplot as plt
plt.hist(data_liping_wine_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
# plot a scatter 
plt.scatter(data_liping_wine_norm['clust'],data_liping_wine_norm['pH'])
plt.scatter(data_liping_wine_norm['clust'],data_liping_wine_norm['chlorides'])

'''
10-	Re-cluster the data into three clusters and check the results. Show the results to your professor.
'''
# Please ref to code line 111-116 
'''
from sklearn.cluster import KMeans
from sklearn import datasets
n_clusters=3    # n_clusters=6
model=KMeans(n_clusters)
model.fit(data_liping_wine_norm)
'''

