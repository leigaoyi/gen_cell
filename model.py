# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:20:14 2019

@author: kasy
"""

import tensorflow as tf

slim = tf.contrib.slim

def generator(noise_in, reuse=False):
    '''
    noise_in = (bs, 64)
    conv_in = (bs, 32, 32, 3)
    out (bs, 64, 64, 3)
    '''
    with tf.variable_scope('generator'):
        fc1 = slim.fully_connected(noise_in, 256)
        fc2 = slim.fully_connected(fc1, 32*32*3)
        conv_shape = tf.reshape(fc2, [fc2.shape[0], 32, 32, 3])
        
        conv_in = slim.conv2d(conv_shape, 32, 5)
        
        down1 = slim.conv2d(conv_in, 64, 3, stride=2)
        conv1 = slim.conv2d(down1, 64, 3)
        conv1 = slim.conv2d(conv1, 64, 3)
        
        down2 = slim.conv2d(conv1, 128, 3, stride=2)
        conv2 = slim.conv2d(down2, 128, 3)
        conv2 = slim.conv2d(conv2, 128, 3)
        
        conv3 = slim.conv2d(conv2, 256, 3)
        conv3 = slim.conv2d(conv3, 256, 3)
        
        up1 = slim.conv2d_transpose(conv3, 256, 3, stride=2)
        concat1 = tf.concat([up1, conv1], axis=3)
        conv4 = slim.conv2d(concat1, 256, 3)
        conv4 = slim.conv2d(conv4, 256, 3)
        
        up2 = slim.conv2d_transpose(conv4, 128, 3, stride=2)
        concat2 = tf.concat([up2, conv_in], axis=3)
        conv5 = slim.conv2d(concat2, 128, 3)
        conv5 = slim.conv2d(conv5, 128, 3)
        
        up3 = slim.conv2d_transpose(conv5, 64, 3, stride=2)
        conv_out = slim.conv2d(up3, 3, 1, activation_fn=tf.nn.sigmoid)
        
        return conv_out
        
def discriminator(img_in, reuse=False):
    
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_in = slim.conv2d(img_in, 64, 5)
        
        conv1 = slim.conv2d(conv_in, 128, 3, stride=2)
        conv1 = slim.conv2d(conv1, 256, 3, stride=2)
        conv1 = slim.conv2d(conv1, 512, 3, stride=2)        
      
        conv2 = slim.conv2d(conv1, 256, 3)
        conv2 = slim.conv2d(conv2, 128, 3, activation_fn=tf.nn.sigmoid)
        return conv2
    
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [20, 32, 32, 3])    
    x_gen = tf.placeholder(tf.float32, [20, 64, 64, 3])
    
    y_gen = generator(x)
    y_dis = discriminator(x_gen)
    print('gen shape', y_gen.shape)
    print('dis shape ', y_dis.shape)