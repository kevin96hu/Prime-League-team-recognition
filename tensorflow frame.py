#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:22:34 2019

@author: kevin
"""
import tensorflow as tf
import numpy as np
#input_data()
x = tf.placeholder(tf.float32,input.shape)
y = tf.placeholder(tf.float32,output.shape)
#----Weight Initialization---#
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#----first convolution layer----#
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#----first maxpooling layer----#
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#----second convolution layer----#
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#----SSP layer----#
def spp_layer(input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):

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
#----full connected layer----# hyperparameter not certain
W_fc1 = weight_variable([, ])
b_fc1 = bias_variable([])
h_fc1 = tf.nn.softmax(tf.matmul(x_flatten,W_fc1) + b_fc1)

#------train and evaluate----#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0],
                                                   y_: batch[1],
                                                   keep_prob: 1.})
            print('setp {},the train accuracy: {}'.format(i, train_accuracy))
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    test_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))
    saver = tf.train.Saver()
    path = saver.save(sess, './my_net/mnist_deep.ckpt')
    print('save path: {}'.format(path))



