 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

without class, CNN with tensorflow, transform tiff to RGB
"""
from skimage.external import tifffile as im
import pandas as pd
import numpy as np
from PIL import Image
import gdal
import scipy as N
import matplotlib.pyplot as pyplot
import math
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
#mnist = input_data.read_data_sets('self.DataSet', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
#    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def FRead_Image_File(file_name=[]):     
    
    RGB = np.zeros((256,256,3),dtype=np.int)
    tif = gdal.Open(file_name)    
    band1 = tif.GetRasterBand(1)
    band2 = tif.GetRasterBand(2)
    band3 = tif.GetRasterBand(3)
    
    red = band1.ReadAsArray()
    green = band2.ReadAsArray()
    blue = band3.ReadAsArray()   
#    gray = (0.299*red + 0.587*green + 0.114*blue)
    RGB[:,:,0] = red
    RGB[:,:,1] = green
    RGB[:,:,2] = blue             
    
    return RGB 

def Get_Batch(indexi):
    Batch_len = 100
    Directory = r'D:\Kaggle\Understanding the Amazon from Space'
    filename = 'One_hot_TraingSet.csv'

    file = pd.read_csv(Directory+'\\'+filename)
    FtrainSetx = file['image_name']
    FtrainSety = file.iloc[:,1:18]
    FMaxTrainLen = FtrainSetx.shape[0]
    FBatch_Current_Index = 100*indexi
    FImage_Folder_Directory = r'D:\Kaggle\Understanding the Amazon from Space\train-tif-v2'  
    FMaxTrainLen = 40479
    

    if FBatch_Current_Index < FMaxTrainLen-Batch_len:
        batch_xs = np.zeros((Batch_len,256,256,3), dtype=np.int)    
        batch_ys = np.zeros((Batch_len,1), dtype = np.int)  
        
        for i in range(FBatch_Current_Index, FBatch_Current_Index+Batch_len):
            Image_File_Name = FtrainSetx.iloc[i]
            Image_File_Directory = FImage_Folder_Directory+'\\'+Image_File_Name + '.tif'
            batch_xs[i-FBatch_Current_Index,:,:,:] = FRead_Image_File(str(Image_File_Directory))
            
        batch_ys = FtrainSety.iloc[FBatch_Current_Index:(FBatch_Current_Index+Batch_len),0:1]        
        FBatch_Current_Index = FBatch_Current_Index + Batch_len
 
    else:
        batch_xs = np.zeros((FMaxTrainLen-FBatch_Current_Index,256,256,3),dtype=np.int)
        batch_ys = np.zeros((FMaxTrainLen-FBatch_Current_Index,1),dtype=np.int) 
        
        for i in range(FBatch_Current_Index,FMaxTrainLen):
            Image_File_Name = FtrainSetx.iloc[i]
            Image_File_Directory = FImage_Folder_Directory+'/'+Image_File_Name+'.tif'  
            # read image files
            batch_xs[i-FBatch_Current_Index,:,:,:] = FRead_Image_File(str(Image_File_Directory))   
        
        batch_ys= FtrainSety.iloc[FBatch_Current_Index:FMaxTrainLen,0:1]
        FBatch_Current_Index = 0
    return batch_xs,batch_ys
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 256,256,3]) # 28x28
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 256, 256, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 128x128x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 64x64x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 32x32x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 16x16x64

## conv2 layer ##
W_conv3 = weight_variable([5, 5, 64, 128]) # patch 5x5, in size 32, out size 64
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 16x16x128
h_pool3 = max_pool_2x2(h_conv2)                                         # output size 8x8x128

## func1 layer ##
W_fc1 = weight_variable([8*8*128, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
                                              
cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = ys))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables()) 


for i in range(100):
    
    batch_xs, batch_ys = Get_Batch(i)
    
#    cost_results = sess.run(cost, feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:1})
#    print("cost before:",cost_results)
    
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
    
#    if i % 50 == 0:
#        print(compute_accuracy(mnist.test.images, mnist.test.labels))    

    cost_results = sess.run(cost, feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:1})
    prediction_result = sess.run(prediction,feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:1})
#    entropy = sess.run(cross_entropy,feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:1})
   
    print("cost after:",cost_results)
    

