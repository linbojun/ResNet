from __future__ import absolute_import
import Util as util
from matplotlib import pyplot as plt


import os
import tensorflow as tf
import numpy as np
import random
import math


class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.batch_size = 1000
        self.in_width = 32
        self.in_height = 32
        self.in_channels = 3
        #self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = 0.002, rho = 0.95)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0015)
        self.loss_list = []

        #structure#
        self.conv_layer1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides=(2, 2), padding="same")
        self.batch_norm = tf.keras.layers.BatchNormalization()

        #do max pool 3x3 with stride 2 here
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same")
        
        #self.res_block1 = residual_block(num_filters = 64)

        #self.res_block2 = residual_block(num_filters = 64)
        self.block1 = make_first_residual_block(64)
        
        
        #self.res_block4 = residual_block(num_filters = 128, downsample = True)

        #self.res_block5 = residual_block(num_filters = 128)
        self.block2 = make_residual_block(128)
        
        #self.res_block8 = residual_block(num_filters = 256, downsample = True)

        #self.res_block9 = residual_block(num_filters = 256)
        self.block3 = make_residual_block(256)
        
        
        #self.res_block14 =residual_block(num_filters = 512, downsample = True)

        #self.res_block15 =residual_block(num_filters = 512)
        self.block4 = make_residual_block(512)
        
        
        #average pool
        self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(num_classes, activation = 'softmax')
        #self.dense2 = tf.keras.layers.Dense(2)




    #@tf.function
    def call(self, inputs):
        token = self.conv_layer1(inputs)
        token = self.batch_norm(token)
        token = tf.nn.relu(token)


        token = self.max_pool(token)
        token = self.block1(token)
        token = self.block2(token)
        token = self.block3(token)
        token = self.block4(token)

        
        
        token = self.average_pool(token)


        logits = self.dense1(token)

        #logits = self.dense2(token)
        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        #print('logits in loss:', logits)
        #print('labels in loss:', labels)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


class residual_block(tf.keras.layers.Layer):
    def __init__(self, num_filters, downsample = False):
        super(residual_block, self).__init__()

        self.num_filters = num_filters
        self.downsample = downsample


        stride = (1, 1)
        if self.downsample == True:
            self.short_cut_layer = tf.keras.Sequential()
            self.short_cut_layer.add( tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 1, padding = 'same', strides = (2,2)) )
            self.short_cut_layer.add( tf.keras.layers.BatchNormalization() )
            stride = (2, 2)
        else:
            self.short_cut_layer = lambda x: x

        self.layer1 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, 
             padding="same", strides = stride)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.layer2 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, 
             padding="same", strides = (1, 1))
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        
    #@tf.function
    def call(self, inputs):
        token = self.layer1(inputs)
        token = self.batch_norm1(token)
        token = tf.nn.relu(token)

        token = self.layer2(token)
        token = self.batch_norm2(token)
        short_cut = self.short_cut_layer(inputs)
        token = tf.nn.relu(tf.keras.layers.add([short_cut, token]))
        return token

def make_residual_block(filter_num):
    res_block = tf.keras.Sequential()
    res_block.add(residual_block(filter_num, downsample = True))
    res_block.add(residual_block(filter_num, downsample = False))
    return res_block

def make_first_residual_block(filter_num):
    res_block = tf.keras.Sequential()
    res_block.add(residual_block(filter_num, downsample = False))
    res_block.add(residual_block(filter_num, downsample = False))
    return res_block
    






