# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:55:39 2019

@author: MX
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 02:27:04 2019

@author: kevin
"""
import os 
import numpy as np
import tensorflow as tf 
from PIL import Image  
 
#initial weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.02)
    return tf.Variable(initial)
#initial bias
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

#convolution layer
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#max_pool layer
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def spp_layer(input_, levels=6, name = 'SPP_layer',pool_type = 'max_pool'):

    '''
    Multiple Level SPP layer.
    
    Works for levels=[1, 2, 3, 6].
    '''
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name):

        for l in range(levels):
        
            l = l + 1
            ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
            
            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1),)
                
            else :
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:

                x_flatten = tf.reshape(pool,(shape[0],-1))
            else:
                x_flatten = tf.concat((x_flatten,pool),axis=1)
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
            

    return x_flatten


x = tf.placeholder(tf.float32, [1,128,128,3])
y_ = tf.placeholder(tf.float32,[1,82])

#first convolution and max_pool layer
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)

#second convolution and max_pool layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_4x4(h_conv2)

#third convolution and max_pool layer
W_conv3 = weight_variable([3,3,64,128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_4x4(h_conv3)

#fourth convolution and max_pool layer
W_conv4 = weight_variable([3,3,128,256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_4x4(h_conv4)

#h_pool4 = spp_layer(h_conv4)

#变成全连接层，用一个MLP处理
reshape = tf.reshape(h_pool4,[1, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,82])
b_fc2 = bias_variable([82])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,r'C:\\Users\\MX\\Desktop\\model3\\.\player.ckpt')

    im=Image.open(r'C:\\Users\\MX\\Desktop\\ozil.jpg')
    im=im.resize((128,128))
    im=np.array(im).astype(np.float32)
    im=np.reshape(im,[-1,128*128*3])
    im=(im-(255/2.0))/255
    x_img=np.reshape(im,[-1,128,128,3])
    output=sess.run(y_conv,feed_dict={x:x_img})
    print(output)











