# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:35:03 2017
# convert raw data into one-hot-dummy data structure
@author: JIN
"""
##Train_Extended.loc[Train_Extended['image_name']=df_train['image_name']]
import skimage
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
##matplotlib inline

pal = sns.color_palette()
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import cv2


os.chdir("D:\\Kaggle\\Understanding the Amazon from Space")
Current_directory= os.getcwd()
print(Current_directory)

## read file information

sample = pd.read_csv('sample_submission_v2.csv')
print(sample.shape)
sample.head()

df_train = pd.read_csv('train_v2.csv')

labels = df_train['tags'].apply(lambda x: x.split(' '))
Train_label = df_train
Train_label.iloc[:,1] = labels

#Train_label_single = Train_label[Train_label['tags']]

from collections import Counter, defaultdict
counts = defaultdict(int)

## definition of training data

for i in labels:
    for j in i:
        counts[j] += 1
lab = list(counts.keys())
lab.append('image_name')
Train_Extended = pd.DataFrame(data=None,columns = lab)
Train_Extended['image_name'] = df_train['image_name']


x=np.zeros(40479)

for i in range(len(Train_Extended.loc[1,:])-1):
    Train_Extended.iloc[:,i]=x

for j in range(len(df_train['image_name'])): 
    tags_temp = list(df_train.iloc[j,1])
    Train_Extended.loc[j, tags_temp]=1
    
##RawImage = pd.DataFrame[data = None, axils = [-1,]]
Train_Extended.to_csv("One_hot_TraingSet.csv")
