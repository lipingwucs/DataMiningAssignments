# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:30:47 2020

@author: Liping
"""

# Visualize 
import matplotlib 
from matplotlib import pyplot as plt
import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling"
filename = 'Customer Churn Model.txt'
fullpath = os.path.join(path,filename)
data_liping=pd.read_csv(fullpath)
data_liping.columns.values

#create a scatterplot
fig_liping = data_liping.plot(kind='scatter',x='Day Mins',y='Day Charge')

# Save the scatter plot
figfilename = "ScatterPlot_Liping.pdf"
figfullpath = os.path.join(path, figfilename)
fig_liping.figure.savefig(figfullpath)


# Plot multiple charts
help(plt.subplot)
import matplotlib.pyplot as plt
figure_liping,axs = plt.subplots(2, 2,sharey=True,sharex=True)
data_liping.plot(kind='scatter',x='Day Mins',y='Day Charge',ax=axs[0][0])
data_liping.plot(kind='scatter',x='Night Mins',y='Night Charge',ax=axs[0][1])
data_liping.plot(kind='scatter',x='Day Calls',y='Day Charge',ax=axs[1][0])
data_liping.plot(kind='scatter',x='Night Calls',y='Night Charge',ax=axs[1][1])

# Plot a histogram
import matplotlib.pyplot as plt
hist_liping= plt.hist(data_liping['Day Calls'],bins=8)
plt.xlabel('Day Calls Value')
plt.ylabel('Frequency')
plt.title('Frequency of Day Calls')

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_liping['Day Calls'])
plt.ylabel('Day Calls')
plt.title('Box Plot of Day Calls')