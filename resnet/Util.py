import tensorflow as tf
import numpy as np
"""
class residual_block(tf.keras.layers.Layer):
    def __init__(self, num_filters, downsample = False):
        super(residual_block, self).__init__()

        self.num_filters = num_filters
        self.downsample = downsample
        self.transform_layer = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 1, padding = 'same', strides = (2,2))
        stride = (1, 1)
        if self.downsample == True:
            stride = (2, 2)

        self.layer1 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, 
             padding="same", strides = stride)
        self.layer2 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, 
             padding="same", strides = (1, 1))
        
    #@tf.function
    def call(self, inputs):
        token = self.layer1(inputs)
        #mean, variance = tf.nn.moments(token, [0,1,2])
        #token = tf.nn.batch_normalization(token, mean, variance, offset = None, scale = None, variance_epsilon = 1e-5)
        token = batch_normal(token)
        token = tf.nn.relu(token)

        token = self.layer2(token)
        #mean, variance = tf.nn.moments(token, [0,1,2])
        #token = tf.nn.batch_normalization(token, mean, variance, offset = None, scale = None, variance_epsilon = 1e-5)
        token = batch_normal(token)
        if self.downsample == False:
            token += inputs
        else:
            token += self.transform_layer(inputs)
        token = tf.nn.relu(token)
        return token

"""
"""
class residual_block(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, downsample = False):
        super(residual_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample
        self.transform_layer_filter = self.add_weight(name = "transform_layer_filter", shape=[1, 1, in_channel, out_channel], trainable=True)
        #self.transform_layer = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 1, padding = 'same', strides = (2,2))

        self.layer1_filter = self.add_weight(name = shape=[3, 3, in_channel, out_channel], trainable=True)
        #self.layer1 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, padding="same", strides = stride)
        
        self.layer2_filter = self.add_weight(shape = [3, 3, out_channel, out_channel], trainable = True)
        #self.layer2 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = 3, padding="same", strides = (1, 1))
        
    #@tf.function
    def call(self, inputs):
        #print("------------------------------")
        #print("num_in_channel:", self.in_channel,", num_out_channel", self.out_channel)
        stride = (1 ,1)
        if self.downsample == True:
            stride = (2, 2)
        token = tf.nn.conv2d(inputs, self.layer1_filter, strides = stride, padding = 'SAME')
        #mean, variance = tf.nn.moments(token, [0,1,2])
        #token = tf.nn.batch_normalization(token, mean, variance, offset = None, scale = None, variance_epsilon = 1e-5)
        token = batch_normal(token)
        token = tf.nn.relu(token)
        #print("In residual_block, token after filter1: ", token.shape)

        token = tf.nn.conv2d(token, self.layer2_filter, strides = (1,1), padding = 'SAME')
        #mean, variance = tf.nn.moments(token, [0,1,2])
        #token = tf.nn.batch_normalization(token, mean, variance, offset = None, scale = None, variance_epsilon = 1e-5)
        token = batch_normal(token)
        #print("In residual_block, token after filter2: ", token.shape)
        if self.downsample == False:
            token += inputs
        else:
            down_sample_inputs = batch_normal(tf.nn.conv2d(inputs, self.transform_layer_filter, strides = (2,2), padding = 'SAME'))
            print("down_sample_inputs shape:", down_sample_inputs.shape)
            token += down_sample_inputs
        token = tf.nn.relu(token)
        #print("output shape", token.shape)
        return token
"""

def residual_block(inputs, num_filters):
    token = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides = (1, 1), padding = "same")(inputs)
    token = tf.keras.layers.BatchNormalization()(token)
    toke = tf.keras.layers.ReLU()(token)
    token = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides = (1, 1), padding = "same")(token)
    token = tf.keras.layers.BatchNormalization()(token)
    #add short cut 
    token += inputs
    token = tf.keras.layers.ReLU()(token)
    return token

def residual_block_downsample(inputs, num_filters):
    token = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides = (2, 2), padding = "same")(inputs)
    token = tf.keras.layers.BatchNormalization()(token)
    toke = tf.keras.layers.ReLU()(token)
    token = tf.keras.layers.Conv2D(num_filters, kernel_size = 3, strides = (1, 1), padding = "same")(token)
    token = tf.keras.layers.BatchNormalization()(token)

    #short cut
    short_cut = tf.keras.layers.Conv2D(num_filters, kernel_size = 1, strides = (2, 2), padding = "same")(inputs)
    short_cut = tf.keras.layers.BatchNormalization()(short_cut)
    token += short_cut
    token = tf.keras.layers.ReLU()(token)
    return token


def batch_normal(token):
    mean, variance = tf.nn.moments(token, [0,1,2])
    token = tf.nn.batch_normalization(token, mean, variance, offset = None, scale = None, variance_epsilon = 1e-5)
    return token

