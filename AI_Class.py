 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
@author: JIN
Class definition with tensorflow
classification: haze, 
Image type: jpg
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.external import tifffile as im
import os

# basic class
class ArtificialIntelligence(object):
    def __init__(object,dimensions):
        pass
    
    def Data_Rearrage(self,dimension2):
        pass     
    
    def SetCurrentDirectory(self,Directory):
        os.chdir(TrainSetDirectory)
        
    def Read_Train_sets(self,Direcoty):
        pass
    
#    define classification class
class Classification(ArtificialIntelligence):
    def __init__(object,dimension):
        pass

#define Regression class    
class Regression(ArtificialIntelligence):
    def __init__(object,dimension):
        pass
# define cluster class
class cluster(ArtificialIntelligence):
    def __init__(object,dimension):
        pass
# define Image classification class
class ImageClassification(Classification):
    # image information
    def __init__(object,dimensions):
        self.xpixls = 256
        self.ypixls = 256
        self.N_Image_Layers = 4
        self.N_TrainSet = N_TrainSet        
        self.Image_Type = Type        
        self.MaxTrainLen = 40479              
# define Image classification class with CNN_tensorflow
class TensorflowCNNImageClassification(ImageClassification):
    def __init__(self,xpixls,ypixls,layers,Max_Samples,Type = ".tif"):
        self.xpixls = xpixls
        self.ypixls = ypixls
        self.N_Image_Layers = layers
        self.MaxTrainlen = Max_Samples
        self.Batch_Current_Index = 0   
        self.Image_Type = Type
        
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
    
    def SetMaxPool(self,ksize,stride):
        # ksize= [1,x_size,y_size,1] 
        #stride = [1,x_stride,y_stride,1]
        if len(ksize) != 4 or len(stride) != 4:
            return False
        else:
            self.MaxPool_ksize = ksize
            self.MaxPool_stride = stride
            return True
    
    def SetConv_Stride(self,stride):
        # stride = [1,x_stride,y_stride,1]
        if len(stride) !=4:
            return False
        else:
            self.Con_Stride = stride
            return True        
    
    def SetConv1(self,conv1_para):
        # conv1_para = [patch_x,patch_y,patch_z,output]
        #patch_x: x pixls of the conv1 patch
        #patch_y: y pixls of the conv2 patch
        #patch_z: z pixls of the conv3 patch
        #output: out put of the convolution
        if len(conv1_para) != 4:
            return False
        else:
            self.Conv1_Parameter = conv1_para
            return True
        
    def SetConv2(self,conv2_para):
        #conv2_para = [patch_x,patch_y,patch_z,output]
        #patch_x: x pixls of thself.Conv2_Parametere conv1 patch
        #patch_y: y pixls of the conv2 patch
        #patch_z: z pixls of the conv3 patch
        #output: out put of the convolution
        if len(conv2_para) != 4:
            return False
        else:
            self.Conv2_Parameter = conv2_para
            return True   
    
    def SetFun1(self,Input_Output):
        # Input_Output = [input_grid,output_grid]
        if len(Input_Output) != 2:
            return False
        else:
            self.Fun1_Input_Output = Input_Output
            return True        
    
    def SetFun2(self,Input_Output):
        # Input_Output = [input_grid,output_grid]
        if len(Input_Output) != 2:
            return False
        else:
            self.Fun2_Input_Output = Input_Output
            return True  
           
    def GetBatchxy(self,Batch_len):
        self.Batch_Train_Setx = []
        self.Batch_Train_Sety = []
        if self.Batch_Current_Index < self.MaxTrainLen-Batch_len:
            self.Batch_Train_Setx = np.zeros((Batch_len,self.xpixls,self.ypixls,self.N_Image_Layers),dtype = np.int32)           
            self.Batch_Train_Sety = np.zeros((Batch_len,self.MaxPredicColumns))  
            
            for i in range(self.Batch_Current_Index, self.Batch_Current_Index+Batch_len):
                Image_File_Name = self.trainSetx.iloc[i]
                Image_File_Directory = self.Image_Folder_Directory+'\\'+Image_File_Name + self.Image_Type
              
                self.Batch_Train_Setx[i-self.Batch_Current_Index,:,:,:] = self.Read_Image_File(str(Image_File_Directory[0]))  
            
            self.Batch_Train_Sety = self.trainSety.iloc[self.Batch_Current_Index:(self.Batch_Current_Index+Batch_len),:]
            self.Batch_Current_Index = self.Batch_Current_Index + Batch_len
            
        else:
            self.Batch_Train_Setx = np.zeros((self.MaxTrainLen-self.Batch_Current_Index,self.xpixls,self.ypixls,self.N_Image_Layers))
            self.Batch_Train_Sety = np.zeros((self.MaxTrainLen-self.Batch_Current_Index,self.MaxPredicColumns))          
            for i in range(self.Batch_Current_Index,self.MaxTrainLen):
                Image_File_Name = self.trainSetx.iloc[i]
                Image_File_Directory = self.Image_Folder_Directory+'/'+Image_File_Name+self.Image_Type  
                # read image files
                self.Batch_Train_Setx[i-self.Batch_Current_Index,:,:,:] = self.Read_Image_File(str(Image_File_Directory[0]))      
                
            self.Batch_Train_Sety= self.trainSety.iloc[self.Batch_Current_Index:self.MaxTrainLen,:]            
            self.Batch_Current_Index = 0
        
    
    def Read_Train_sets(self,Directory,filename,Train_Columns,Predict_Column):
        file = pd.read_csv(Directory+'/'+filename)
        self.trainSetx = file[Train_Columns]
        self.trainSety = file[Predict_Column]
        self.MaxTrainLen = self.trainSetx.shape[0]
        self.MaxPredicColumns = self.trainSetx.shape[1]
        
    def Read_Image_File(self,file_name=[]):
        
        if self.Image_Type == '.tif':
            return im.imread(file_name)
        
        elif self.Image_Type =='.jpg':
            return             
        
        elif self.Image_Type == '.png':
            return
        
        elif self.Image_Type == '':
            return
        
        
    def DataValidation(self):
        pass
    
    def Input_Output_Define(self,Input_dimension = [], Output_dimension = []):
        #input = [None, 256, 256,4] 256X256X4
        #output = [None, 1] output is a digit
        self.xs = tf.placeholder(tf.float32, Input_dimension)
        self.ys = tf.placeholder(tf.float32, Output_dimension)
        self.keep_prob = tf.placeholder(tf.float32)     
        
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, dtype = tf.float32, stddev=0.1) 
#        initial = tf.constant(0.000001, dtype = tf.float32, shape=shape)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.01, dtype = tf.float32, shape=shape)
        return tf.Variable(initial)
    
    def max_pool(self,x):
        return tf.nn.max_pool(x, self.MaxPool_ksize,self.MaxPool_stride, padding='SAME')
    
    def Conv1Layer(self,W_conv1_matrix = []):       
        self.W_conv1 = self.weight_variable(self.Conv1_Parameter) # patch 2x2, in size 4, out size 32
        self.b_conv1 = self.bias_variable([self.Conv1_Parameter[len(self.Conv1_Parameter)-1]])
        self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.xs, self.W_conv1, self.Con_Stride, padding='SAME') + self.b_conv1)# output size 64X64X32
        self.h_conv1_size = []
        self.h_pool1 = self.max_pool(self.h_conv1)   # output size 16X16x32
        self.h_pool1_size = []
        
    def Conv2Layer(self):
        self.W_conv2 = self.weight_variable(self.Conv2_Parameter) # patch 2x2, in size 32, out size 128
        self.b_conv2 = self.bias_variable([self.Conv2_Parameter[len(self.Conv2_Parameter)-1]])
        self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_pool1, self.W_conv2, self.Con_Stride, padding='SAME') + self.b_conv2)# output size 4x4x128
        self.h_pool2 = self.max_pool(self.h_conv2)   # output size 1x1x128
        self.h_pool2_size = []
        self.h_conv2_size = []   
        
    def Function1Layer(self):
        reshape_factor = 1
#        for i in range(1,len(self.h_pool2.shape)):
#            reshape_factor = reshape_factor*self.h_pool2.shape[i]
#        
#        if reshape_factor != len(self.Fun1_Input_Output(1)):
#            print("error: func1 parameters incompatible")
            
        self.W_fc1 = self.weight_variable(self.Fun1_Input_Output)
        self.b_fc1 = self.bias_variable([self.Fun1_Input_Output[1]])        
#        h_pool2_flat = tf.reshape(self.h_pool2, [-1,64*64*128]) 
            
        self.h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(self.h_pool2, [-1,self.Fun1_Input_Output[0]]), self.W_fc1) + self.b_fc1)        
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        
    def Function2Layer(self):
        self.W_fc2 = self.weight_variable(self.Fun2_Input_Output)
        self.b_fc2 = self.bias_variable([self.Fun2_Input_Output[1]])
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)  
#        self.prediction = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
    
    def compute_accuracy(self):
#        y = self.Batch_Train_Sety['haze']
        y_pre = self.sess.run(self.prediction, feed_dict={self.xs: self.Batch_Train_Setx, self.keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(self.ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.xs: self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob: 1})
        return result
        
    def CommitteTraining(self,steps,run_times = 1000,saveFileName = 'Model'):
        ##self.GetBatchxy(100) 
        self.Conv1Layer()
        self.Conv2Layer()
        self.Function1Layer()
        self.Function2Layer()
        
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction),reduction_indices=[1]))       # loss

        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction),reduction_indices=[1]))
        
        self.cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction, labels = self.ys))
         
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cost)
        
        self.sess = tf.Session()
#        self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

        # important step            
        for _ in range(run_times):            
            
            self.GetBatchxy(100)
            cost = self.sess.run(self.cost, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1})
            print("iteration: cost",_,cost)
            
            cost1 = self.sess.run(self.train_step, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1})
#            error = tf.square(self.Batch_Train_Sety-self.prediction)

#            prediction = self.sess.run(self.prediction, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1})
#            MSE = self.sess.run(self.prediction, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1})
            cost = self.sess.run(self.cost, feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1})
            print("iteration: cost",_,cost)
            
#                cross_entropy = self.sess.run(self.cross_entropy,feed_dict={self.xs:self.Batch_Train_Setx, self.ys:self.Batch_Train_Sety, self.keep_prob:1} )
#                print("entropy",cross_entropy)

            
        saver = tf.train.Saver()
        saver.save(self.sess,saveFileName) 
        
    def Predict(self,Image_path,Save_Path):
        
        batch_prediction = 100
        file_list = os.listdir(Image_path)
        
        self.results = np.zeros((len(file_list), self.MaxPredicColumns), dtype = 'float32')
        self.test_image = np.zeros((1000,self.xpixls,self.ypixls,self.N_Image_Layers),dtype = 'float32')
        
        #for i in range(int(len(file_list)): 
            
        for i in range(1000): 
            file_name = Image_path+'\\'+file_list[i]
            self.test_image[i,:,:,:] = self.Read_Image_File(file_name)
        
        y_pre = self.sess.run(self.prediction, feed_dict={self.xs: self.test_image, self.keep_prob:0.5})            
        self.results = y_pre[:,0]
        print(y_pre)
            
        self.results = pd.DataFrame(self.results)
        self.results.to_csv(Save_Path)

        
    def Close_session(self):
        self.sess.close();        
        
            

# initializing
ImageInstance = TensorflowCNNImageClassification(256,256,4,40479)
#set  TrainingSet Directory
ImageInstance.Set_TrainSet_Directory(r'D:\Kaggle\Understanding the Amazon from Space\Split')
#set Image Files Directory
ImageInstance.Set_Image_Directory(r'D:\Kaggle\Understanding the Amazon from Space\train-tif-v2')
# Read Training Sets
ImageInstance.Read_Train_sets(r'D:\Kaggle\Understanding the Amazon from Space\Split','haze.csv',['image_name'],['haze'])

# Define Conv1 layer
ImageInstance.Input_Output_Define([None, 256, 256,4],[None, 1])#256X256X4

ImageInstance.SetConv_Stride([1,1,1,1])

ImageInstance.SetMaxPool([1,16,16,1],[1,4,4,1])
# Define Conv1 layer
ImageInstance.SetConv1([2,2,4,32]) # 64X64X32

ImageInstance.SetConv2([4, 4, 32, 32]) #16X16X32
# define Function1 layer
ImageInstance.SetFun1([32*16*16, 16])
# define Function2 Layer
ImageInstance.SetFun2([16, 1])


ImageInstance.CommitteTraining(0.1,run_times = 20,saveFileName = r"D:\Kaggle\Understanding the Amazon from Space\Split\haze")

#ImageInstance.Predict(r"D:\Kaggle\Understanding the Amazon from Space\test-tif-v2",r"D:\Kaggle\Understanding the Amazon from Space\Split\haze")


#ImageInstance.Close_session()

