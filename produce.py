# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:48:46 2019

@author: robot
"""

import tensorflow as tf
import os
from model import generator
import numpy as np
from skimage import io

from tqdm import tqdm

produce_num = 1000

result_dir = './result/produce/'

ckpt_path = './checkpoints/cell_fig_400.ckpt'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
x = tf.placeholder(tf.float32, [1, 100])    
y = generator(x)

sess = tf.Session()
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess.run(init)
saver.restore(sess, ckpt_path)

for i in tqdm(range(produce_num)):
    noise = np.random.uniform(-1, 1, size=(1, 100))
    feed_dice = {x:noise}
    fig = sess.run(y, feed_dict=feed_dice)
    fig = np.reshape(fig, (64, 64, 3))
    io.imsave(result_dir+'{0}.jpg'.format(i), fig)