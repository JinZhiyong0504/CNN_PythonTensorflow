# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:46:17 2017

@author: JIN
"""


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.external import tifffile as im
import os
   
    
    
train_set = pd.read_csv(r'D:\Kaggle\Understanding the Amazon from Space\One_hot_TraingSet.csv')
print(train_set.shape)
train_set = pd.DataFrame(train_set)

Primary = train_set[['primary','image_name']]
Primary.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\primary.csv')

Haze = train_set[['haze','image_name']]
Haze.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\haze.csv')

agriculture = train_set[['agriculture','image_name']]
agriculture.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\agriculture.csv')

clear = train_set[['clear','image_name']]
clear.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\clear.csv')

water = train_set[['water','image_name']]
water.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\water.csv')

habitation = train_set[['habitation','image_name']]
habitation.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\habitation.csv')


road = train_set[['road','image_name']]
road.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\road.csv')

cultivation = train_set[['cultivation','image_name']]
cultivation.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\cultivation.csv')

slash_burn = train_set[['slash_burn','image_name']]
slash_burn.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\slash_burn.csv')

cloudy = train_set[['cloudy','image_name']]
cloudy.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\cloudy.csv')

partly_cloudy = train_set[['partly_cloudy','image_name']]
partly_cloudy.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\partly_cloudy.csv')

conventional_mine = train_set[['conventional_mine','image_name']]
conventional_mine.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\conventional_mine.csv')

bare_ground = train_set[['bare_ground','image_name']]
bare_ground.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\bare_ground.csv')

water = train_set[['water','image_name']]
water.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\water.csv')

artisinal_mine = train_set[['artisinal_mine','image_name']]
artisinal_mine.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\artisinal_mine.csv')

blooming = train_set[['blooming','image_name']]
blooming.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\blooming.csv')

selective_logging = train_set[['selective_logging','image_name']]
selective_logging.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\selective_logging.csv')

blow_down = train_set[['blow_down','image_name']]
blow_down.to_csv(r'D:\Kaggle\Understanding the Amazon from Space\Split\blow_down.csv')