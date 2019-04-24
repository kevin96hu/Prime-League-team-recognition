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
 
cwd=r'C:\\Users\\MX\\Desktop\\photoface\\'
classes=os.listdir(r'C:\\Users\\MX\\Desktop\\photoface')  #人为设定2类
#writer= tf.python_io.TFRecordWriter("dog_and_cat_train.tfrecords") #要生成的文件
writer= tf.python_io.TFRecordWriter(r'C:\\Users\\MX\\Desktop\\record\\cnn.tfrecords') #要生成的文件
i = 0
for index,name in enumerate(classes):
    class_path=cwd+name+'\\'
    for img_name in os.listdir(class_path): 
        if img_name=='.DS_Store':
            pass
        else:
            img_path=class_path+img_name #每一个图片的地址
     
            img=Image.open(img_path)
            i += 1
            img_raw=img.tobytes()#将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串
 
writer.close()

def read_and_decode(filename): # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename],shuffle=True)#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, tf.stack([128, 128, 3]))  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.float32) #在流中抛出label张量
 
    return img, label

epoch = 3
batch_size = 64

def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

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

def spp_layer(input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):

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


x = tf.placeholder(tf.float32, [batch_size,128,128,3])
y_ = tf.placeholder(tf.float32, [batch_size,82])

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
#h_pool4 = max_pool_4x4(h_conv4)

h_pool4 = spp_layer(h_conv4)

#变成全连接层，用一个MLP处理
reshape = tf.reshape(h_pool4,[batch_size, -1])
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

#损失函数及优化算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


img, label = read_and_decode(r'C:\\Users\\MX\\Desktop\\record\\cnn.tfrecords')
#img_test, label_test = ReadMyOwnData.read_and_decode("dog_and_cat_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
#img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
#                                                batch_size=batch_size, capacity=2000,
#                                                min_after_dequeue=1000)
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
    batch_idxs = int(2795/batch_size)
    for i in range(epoch):
        for j in range(batch_idxs):
            val, l = sess.run([img_batch, label_batch])
            l = one_hot(l,82)
            _, acc = sess.run([train_step, accuracy], feed_dict={x: val, y_: l, keep_prob: 0.8})
            print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i, j, batch_idxs, acc) )
    '''val, l = sess.run([img_test, label_test])
    l = one_hot(l,2)
    print(l)
    y, acc = sess.run([y_conv,accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
    print(y)
    print("test accuracy: [%.8f]" % (acc))'''
    saver.save(sess,r'C:\\Users\\MX\\Desktop\\model3\\player.ckpt')

    coord.request_stop()
    coord.join(threads)













