# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:39:30 2020

@author: Liping

"""

#Generate one number between 1 and 100
import numpy as np
np.random.randint(1,100)

#Generate a random number between 0 and 1
import numpy as np
np.random.random()


#Define a function to generate several random numbers in a range
def randint_range_liping(n,a,b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x
n=10
a=30
b=70
list_x= randint_range_liping(n,a,b)
print(list_x)

#d.	Generate three random numbers between 0 and 100, which are all multiples of 5
import random
for i in range(3):
    print( random.randrange(0,100,5)) 
    
#  Select three numbers randomly from a list of numbers 
    list = [20, 30, 40, 50 ,60, 70, 80, 90]
sampling = random.choices(list, k=3)
print("sampling with choices method ", sampling)

#Generate a set of random numbers that retain their value, i.e. use the seed option
np.random.seed(1)
for i in range(3):
    print (np.random.random())
    
#Shuffle a list of 5 numbers
a = [1,2,3,4,5]
print(a)
np.random.shuffle(a)    
print(a)


import pandas as pd
import os
filepath="D:/CentennialWu/2020Fall/COMP309Data/Assignments/Lab06DataLoading&Wrangling/lotofdata"
filename ="001.csv"
fullpath = os.path.join(filepath,filename)
data_final=pd.read_csv(fullpath)

data_final_size=len(data_final)
print(data_final_size)
for i in range(1,333):
    if i<10:
        filename='0'+'0'+str(i)+'.csv'
    if 10<=i<100:
        filename='0'+str(i)+'.csv'
    if i>=100:
        filename=str(i)+'.csv'  
        
    file=filepath+'/'+filename
    #print(file)        
    data=pd.read_csv(file)
    data_final_size+=len(data)
    #print(data_final_size)
    data_final=pd.concat([data_final,data],axis=0)
print (data_final_size)
data_final.shape



