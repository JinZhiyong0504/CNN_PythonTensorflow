 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
@author: JIN
Data source Kaggel Amazon Image
classification: Primary, 
Image type: jpg
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.external import tifffile as im
import os

# number 1 to 10 data
class ArtificialIntelligence(object):
    def __init__(object,dimensions):
        pass
    
    def Data_rearrage(self,dimension2):
        pass     
    
    def SetCurrentDirectory(self,Directory):
        os.chdir(TrainSetDirectory)
        
    def Read_Train_sets(self,Direcoty):
        pass
    
class Classification(ArtificialIntelligence):
    def __init__(object,dimension):
        pass
    
class Regression(ArtificialIntelligence):
    def __init__(object,dimension):
        pass
    
class cluster(artificialIntelligence):
    def __init__(object,dimension):
        pass
    
class ImageClassification(Classification):
    # image information
    def __init__(object,dimensions):
        self.xpixls = 256
        self.ypixls = 256
        self.N_Image_Layers = 4
        self.N_TrainSet = N_TrainSet        
        self.Image_Type = Type        
        self.MaxTrainLen = 40479
    
        
        
        

class TensorflowCNNImageClassification(ArtificialIntelligence):
    def __init__(self,xpixls,ypixls,layers,N_TrainSet,Type = ".tif"):
        self.DataSetx = None
        self.DataSety = []
        self.MaxTrainlen = [] 
        self.Batch_Current_Index = 0       

        

        
        
        
        ## image folder Directories and Trainset file directory
        self.TrainSet_Directory = []
        self.Image_Directory = []
        self.Strid = [1,1,1,1]
        
        ## TrainSet CSV file information
        self.TrainSet_Columns_Begin = 0
        self.TrainSet_Columns_End = 0
        self.TrainSet_FileName_Columns = 18
        self.MaxPredicColumns = 1
        
        ##training parameters
        self.keep_prob = 0.5        
        
    def Set_Image_Directory(self,Image_Directory):
        self.Image_Folder_Directory = Image_Directory
        
    def Set_TrainSet_Directory(self,TrainSetDirectory):
        self.TrainSet_Directory = TrainSetDirectory
        os.chdir(TrainSetDirectory)

            
    def GetBatchxy(self,Batch_len):
        self.Batch_Train_Setx = []
        self.Batch_Train_Sety = []
        if self.Batch_Current_Index < self.MaxTrainLen-Batch_len:
            self.Batch_Train_Setx = np.zeros((Batch_len,self.xpixls,self.ypixls,self.N_Image_Layers),dtype = 'f')           
            self.Batch_Train_Sety = np.zeros((Batch_len,self.MaxPredicColumns))  
            
            for i in range(self.Batch_Current_Index, self.Batch_Current_Index+Batch_len):
                Image_File_Name = self.trainSet.loc[i,'image_name']
                Image_File_Director = self.Image_Folder_Directory+'/'+Image_File_Name + self.Image_Type
                self.Batch_Train_Setx[i-self.Batch_Current_Index,:,:,:] = im.imread(Image_File_Director)   
            
            self.Batch_Train_Sety = self.trainSet.iloc[self.Batch_Current_Index:(self.Batch_Current_Index+Batch_len),1:18]
            self.Batch_Current_Index = self.Batch_Current_Index + Batch_len
            
        else:
            self.Batch_Train_Setx = np.zeros((self.MaxTrainLen-self.Batch_Current_Index,self.xpixls,self.ypixls,self.N_Image_Layers))
            self.Batch_Train_Sety = np.zeros((self.MaxTrainLen-self.Batch_Current_Index,self.MaxPredicColumns))          
            for i in range(self.Batch_Current_Index,self.MaxTrainLen):
                Image_File_Name = self.trainSet.loc[i,'image_name']
                Image_File_Director = self.Image_Folder_Directory+'/'+Image_File_Name+self.Image_Type  
                # read image files
                self.Batch_Train_Setx[i-self.Batch_Current_Index,:,:,:] = im.imread(Image_File_Director)       
                
            self.Batch_Train_Sety= self.trainSet.iloc[self.Batch_Current_Index:self.MaxTrainLen-Batch_len,1:18]
            
            self.Batch_Current_Index = 0
        
    
    def Read_Train_sets(self,Image_Directory,filename,Pred_StartIndex,Pred_EndIndex,FileName_Columns):
        
        self.trainSet = pd.read_csv(Image_Directory+'/'+filename)
        self.MaxTrainLen = len(self.trainSet.index)         
        self.TrainSet_Columns_Begin = Pred_StartIndex
        self.TrainSet_Columns_End = Pred_EndIndex
        self.TrainSet_FileName_Columns = FileName_Columns
        self.MaxPredicColumns = self.TrainSet_Columns_End-self.TrainSet_Columns_Begin+1

    def SetData(self,xpixls = 0, ypixls = 0, layers = 0, yDimensions =0, Records=0):
        self.MaxTrainLen = len(Records)
        self.Input_Output_Define(xpixls*ypixls*layers,yDimensions)
    
    def DataValidation(self):
        pass
        
    def SetNuralNet(self,):
        pass
        
#    def rearrange(self,xpixls,ypixls,Image_layers):
#         x_image = tf.reshape(self.Batch_Train_Setx, [-1, xpixls*Image_layer*ypixls, 1])
    
    def Input_Output_Define(self,Input_dimension = 0, Output_dimension = 0):
        self.xs = tf.placeholder(tf.float16, [None, 256, 256,4]) # 256X256X4
        self.ys = tf.placeholder(tf.float16, [None, 1])
        self.keep_prob = tf.placeholder(tf.float16)
          
        
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, dtype = tf.float16, stddev=0.1) 
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1, dtype = tf.float16, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self,x, W):
        print(W)
        return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')
    
    def max_pool(self,x):
        return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    
    def Conv1Layer(self,W_conv1_matrix = []):
        
        self.W_conv1 = self.weight_variable([4,4,4,32]) # patch 2x2, in size 4, out size 32
        self.b_conv1 = self.bias_variable([32])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.xs, self.W_conv1) + self.b_conv1) # output size 64X64X32
        self.h_conv1_size = []
        self.h_pool1 = self.max_pool(self.h_conv1)   # output size 16X16x32
        self.h_pool1_size = []
        
    def Conv2Layer(self):
        self.W_conv2 = self.weight_variable([4, 4, 32, 128]) # patch 2x2, in size 32, out size 128
        self.b_conv2 = self.bias_variable([128])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) # output size 4x4x128
        self.h_pool2 = self.max_pool(self.h_conv2)   # output size 1x1x128
        self.h_pool2_size = []
        self.h_conv2_size = []
        
    def compute_accuracy(self):
        y = self.Batch_Train_Sety['primary']
        y_pre = self.sess.run(self.prediction, feed_dict={self.xs: self.Batch_Train_Setx, self.keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(self.ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.xs: self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety.iloc[:,0:1], self.keep_prob: 1})
        return result
    
        
    def Function1Layer(self):
        self.W_fc1 = self.weight_variable([128, 512])
        self.b_fc1 = self.bias_variable([512])        
#        h_pool2_flat = tf.reshape(self.h_pool2, [-1,64*64*128])
        self.h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(self.h_pool2, [-1,128]), self.W_fc1) + self.b_fc1)        
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
    def Function2Layer(self):
        self.W_fc2 = self.weight_variable([512, 1])
        self.b_fc2 = self.bias_variable([1])
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)        
        
    def CommitteTraining(self,steps,saveFileName = 'Model'):
        ##self.GetBatchxy(100) 
        self.Input_Output_Define(256*256*4,1)
        self.Conv1Layer()
        self.Conv2Layer()
        self.Function1Layer()
        self.Function2Layer()
        
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction),reduction_indices=[1]))       # loss

        train_step = tf.train.AdamOptimizer(steps).minimize(self.cross_entropy)
        
        with tf.Session() as self.sess:
            self.sess.run( tf.initialize_all_variables() )

            # important step            
            for _ in range(1000):

                self.GetBatchxy(100)
                #self.Batch_Train_Setx = tf.reshape(self.Batch_Train_Setx, [-1, 256, 256, 4])
                #self.Batch_Train_Setx = np.float16(self.Batch_Train_Setx)
                #self.Batch_Train_Sety = np.float16(self.Batch_Train_Sety)
                #batch_xs, batch_ys = self.DataSet.train.next_batch(100)
#                primary = self.Batch_Train_Sety[:,1]
#                self.Batch_Train = []
#                self.Batch_Train_Sety = primary
          
                self.sess.run(train_step, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety.iloc[:,0:1], self.keep_prob:0.5})
                
            saver = tf.train.Saver()
            saver.save(self.sess,saveFileName)       
            

# initializing
ImageInstance = TensorflowCNNImageClassification(256,256,4,40479)
#set current Direcory

#set  TrainingSet Directory
ImageInstance.Set_TrainSet_Directory(r'D:\Kaggle\Understanding the Amazon from Space')
#set Image Files Directory
ImageInstance.Set_Image_Directory(r'D:\Kaggle\Understanding the Amazon from Space\train-tif-v2')
# Read Training Sets
ImageInstance.Read_Train_sets(r'D:\Kaggle\Understanding the Amazon from Space','One_hot_TraingSet.csv',1,1,18)

# Define Conv1 layer

# Define Conv2 layer

# define Function1 layer


# define Function2 Layer


# commmit trainig and saving after training

ImageInstance.CommitteTraining(0.0001)







#ImageInstance.Input_Output_Define(256*256*4,1)
#ImageInstance.Conv1Layer()
#ImageInstance.Conv2Layer()
#ImageInstance.Function1Layer()
#ImageInstance.Function2Layer()
#ImageInstance.GetBatchxy(1000)
#ImageInstance.GetBatchxy(100)
