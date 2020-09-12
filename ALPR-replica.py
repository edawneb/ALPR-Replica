# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:03:55 2020

@author: bentw
"""


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, MaxPooling3D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


model1 = Sequential([
    Conv2D(16, (3,3), activation = tf.nn.leaky_relu),
    MaxPooling3D(2, 2, 2),
    Conv2D(32, (3, 3), activation = tf.nn.leaky_relu),
    MaxPooling3D(2, 2, 2),
    Conv2D(64, (3, 3), activation = tf.nn.leaky_relu),
    MaxPooling3D((2, 2, 2)),
    Conv2D(128, (3, 3), activation = tf.nn.leaky_relu),    
    MaxPooling3D((2, 2, 2)),    
    Conv2D(256, (3, 3), activation = tf.nn.leaky_relu),
    MaxPooling3D((2, 2, 2)),    
    Conv2D(512, (3, 3), activation = tf.nn.leaky_relu),
    MaxPooling3D((2, 2, 1)),    
    Conv2D(1024, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(1024, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(30, (1, 1), activation = 'linear'),
    #Conv2D(35, (1, 1), activation = 'linear')
    ])
##instructions showed 30/35


model2 = Sequential([
    Conv2D(32, (3,3), activation = tf.nn.leaky_relu),
    Conv2D(64, (3, 3), activation = tf.nn.leaky_rely),
    Conv2D(128, (3, 3), activation = tf.nn.leaky_reul),
    Conv2D(64, (1, 1), activation = tf.nn.leaky_relu),
    Conv2D(128, (3,3), activation = tf.nn.leaky_relu),
    Conv2D(256, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(128, (1, 1), activation = tf.nn.leaky_relu),
    Conv2D(256, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(512, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(256, (1, 1), activation = tf.nn.leaky_relu),
    Conv2D(512, (3, 3), activation = tf.nn.leaky_relu),
    Conv2D(30, (1, 1), activation = 'linear'),
    
    ])