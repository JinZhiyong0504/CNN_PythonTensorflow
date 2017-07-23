"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
## number 1 to 10 data
#mnist = input_data.read_data_sets('FDataSet', one_hot=True)
from skimage.external import tifffile as im
import pandas as pd
import numpy as np
from PIL import Image
import gdal
import scipy as N
import matplotlib.pyplot as pyplot


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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME')

def FRead_Image_File(file_name=[]):
#    RGB_primary = np.zeros((256,256,3))
    
#    tif_primary = gdal.Open(r"D:\Kaggle\Understanding the Amazon from Space\train-tif-v2\train_0.tif")
#    band1 = tif_primary.GetRasterBand(1)
#    band2 = tif_primary.GetRasterBand(2)
#    band3 = tif_primary.GetRasterBand(3)
#    
#    red = band1.ReadAsArray()
#    green = band2.ReadAsArray()
#    blue = band3.ReadAsArray()   
#    
#    RGB_primary[:,:,0] = red
#    RGB_primary[:,:,1] = green
#    RGB_primary[:,:,2] = blue   
    
    RGB = np.zeros((256,256,3))
    tif = gdal.Open(file_name)    
    band1 = tif.GetRasterBand(1)
    band2 = tif.GetRasterBand(2)
    band3 = tif.GetRasterBand(3)
    
    red = band1.ReadAsArray()
    green = band2.ReadAsArray()
    blue = band3.ReadAsArray()   
    gray = (0.299*red + 0.587*green + 0.114*blue)
    RGB[:,:,0] = red
    RGB[:,:,1] = green
    RGB[:,:,2] = blue     
    
   
    return RGB

Directory = r'D:\Kaggle\Understanding the Amazon from Space'
filename = 'One_hot_TraingSet.csv'

file = pd.read_csv(Directory+'\\'+filename)
FtrainSetx = file['image_name']
FtrainSety = file.iloc[:,1:18]
        

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 256,256,3]) # 256X256X4
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 256, 256,3])
# print(x_image.shape)  # [n_samples, 256,256,4]

## conv1 layer ##
W_conv1 = weight_variable([16, 16, 3, 32]) 
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 16X16X32

## conv2 layer ##
W_conv2 = weight_variable([16, 16, 32, 128]) # patch 
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 1X1X128

## func1 layer ##
W_fc1 = weight_variable([128, 256])
b_fc1 = bias_variable([256])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([256, 1])
b_fc2 = bias_variable([1])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
                                              
cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = ys))

train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer()) 

Directory = r'D:\Kaggle\Understanding the Amazon from Space'
filename = 'One_hot_TraingSet.csv'

file = pd.read_csv(Directory+'\\'+filename)
FtrainSetx = file['image_name']
FtrainSety = file.iloc[:,1:18]
FMaxTrainLen = FtrainSetx.shape[0]

FImage_Folder_Directory = r'D:\Kaggle\Understanding the Amazon from Space\train-tif-v2'  
FMaxTrainLen = 40479
FBatch_Current_Index =0


for i in range(100):
    
    batch_xs = []
    batch_ys = []
    
    Batch_len = 100
    if FBatch_Current_Index < FMaxTrainLen-Batch_len:
        batch_xs = np.zeros((Batch_len,256,256,3))    
        batch_ys = np.zeros((Batch_len,17))  
        for i in range(FBatch_Current_Index, FBatch_Current_Index+Batch_len):
            Image_File_Name = FtrainSetx.iloc[i]
            Image_File_Directory = FImage_Folder_Directory+'\\'+Image_File_Name + '.tif'
            batch_xs[i-FBatch_Current_Index,:,:,:] = FRead_Image_File(str(Image_File_Directory)) 
            
        batch_ys = FtrainSety.iloc[FBatch_Current_Index:(FBatch_Current_Index+Batch_len),2]        
        FBatch_Current_Index = FBatch_Current_Index + Batch_len
 
    else:
        batch_xs = np.zeros((FMaxTrainLen-FBatch_Current_Index,256,256,3))
        batch_ys = np.zeros((FMaxTrainLen-FBatch_Current_Index,1)) 
        
        for i in range(FBatch_Current_Index,FMaxTrainLen):
            Image_File_Name = FtrainSetx.iloc[i]
            Image_File_Directory = FImage_Folder_Directory+'/'+Image_File_Name+'.tif'  
            # read image files
            batch_xs[i-FBatch_Current_Index,:,:,:] = FRead_Image_File(str(Image_File_Directory))    
        
        batch_ys= FtrainSety.iloc[FBatch_Current_Index:FMaxTrainLen,1]
        FBatch_Current_Index = 0
    
    batch_xs = batch_xs
        
    batch_ys = np.reshape(batch_ys,[-1,1])
    
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})    
    cost_results = sess.run(cost, feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:2})    
#    if i % 50 == 0:
#        print(compute_accuracy(mnist.test.images, mnist.test.labels))    


#    entropy = sess.run(cross_entropy,feed_dict={xs: batch_xs, ys:batch_ys , keep_prob:1})
   
    print("cost after:",cost_results)

