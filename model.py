# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:20:14 2019

@author: kasy
"""

import tensorflow as tf

slim = tf.contrib.slim

def generator(noise_in, reuse=False):
    '''
    noise_in = (bs, 100)
    conv_in = (bs, 32, 32, 3)
    out (bs, 64, 64, 3)
    '''
    with tf.variable_scope('generator_mnist', reuse=reuse):
        fc1 = slim.fully_connected(noise_in, 4*4*32)
        reshape = tf.reshape(fc1, [fc1.shape[0], 4, 4, 32])
        
        deconv1 = slim.conv2d_transpose(reshape, 64, 3, stride=2)#8
        deconv2 = slim.conv2d_transpose(deconv1, 128, 3, stride=2)
        #deconv2 = deconv2[:, :7, :7, :]
        
        deconv3 = slim.conv2d_transpose(deconv2, 256, 3, stride=2)
        deconv4 = slim.conv2d_transpose(deconv3, 3, 3, stride=2, \
                                        activation_fn=tf.nn.tanh)
    return deconv4 
        
def discriminator(img_in, reuse=False):
    
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_in = slim.conv2d(img_in, 64, 5)        
        conv1 = slim.conv2d(conv_in, 128, 3, stride=2)
        conv1 = slim.conv2d(conv1, 256, 3, stride=2)
        conv2 = slim.conv2d(conv1, 128, 1)
        reshape = tf.reshape(conv2, [conv2.shape[0], -1])
        
        fc = slim.fully_connected(reshape, 64, activation_fn=None)
        return fc
    
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [20, 100])    
    x_gen = tf.placeholder(tf.float32, [20, 28, 28, 1])
    
    y_gen = generator(x)
    y_dis = discriminator(x_gen)
    print('gen shape', y_gen.shape)
    print('dis shape ', y_dis.shape)