# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:26:17 2017

@author: JIN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

x_data = np.random.rand(100).astype(np.float32).reshape(10,10)
y_data = x_data*0.1 +0.3

## create tensorflow structure start##
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

## creat tensorflow structure end##

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
