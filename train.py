# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:20:44 2019

@author: kasy
"""

import tensorflow as tf
import numpy as np


from tqdm import tqdm
import os
from skimage import io
from opt import save_fig

from util import process_data
from model import generator, discriminator

batch_size = 25
epoch = 400
start_epoch = 0

result_dir = './result/'
ckpt_dir = './checkpoints/'
ckpt_path = ckpt_dir + 'cell_fig_300.ckpt'
ckpt_path = '0'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)    

#===============loss function==========  
z = tf.placeholder(tf.float32, [batch_size, 100])    
x = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])

x_model = generator(z, reuse=False)
y_fake = discriminator(x_model, reuse=False)
y_true = discriminator(x, reuse=True)


recon_loss = tf.reduce_mean(tf.abs(x_model - x))
gen_1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_fake, labels=tf.ones_like(y_fake)))

dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_fake, labels=tf.zeros_like(y_fake)))

dis_loss_true=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_true, labels=tf.ones_like(y_true)))

dis_loss = dis_loss_fake + dis_loss_true        
gen_loss = gen_1_loss

#=====================optimization================
var_list = tf.trainable_variables()
gen_var = [x for x in var_list if 'gen' in x.name]
dis_var = [x for x in var_list if 'dis' in x.name]

opt_g = tf.train.AdamOptimizer(0.0002, beta1=0.5)
opt_g = opt_g.minimize(gen_loss, var_list=gen_var)

opt_d = tf.train.AdamOptimizer(0.0002, beta1=0.5)
opt_d = opt_d.minimize(dis_loss, var_list=dis_var)

#=====================training====================
num_input = len(process_data)
batch_idx = num_input//batch_size

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(max_to_keep=8)

#==========restore if necessary=======
if os.path.exists(ckpt_path+'.index'):
    saver.restore(sess, ckpt_path)
    start_epoch = 300

for i in tqdm(range(start_epoch, epoch)):
    
    for j in range(batch_idx):
        batch_data = process_data[j*batch_size:(j+1)*batch_size]
        batch_data = np.reshape(batch_data, [batch_size, 64, 64, 3])
#        batch_data = process_data[j*batch_size:(j+1)*batch_size]
#        batch_data = np.reshape(batch_data, [batch_size, 28, 28, 1])
        
        batch_noise = np.random.uniform(-1, 1, size=[batch_size, 100])
        
        feed_dict = {z:batch_noise, x:batch_data}
        sess.run(opt_d, feed_dict=feed_dict)
        sess.run(opt_g, feed_dict=feed_dict)
        
        gen_loss_val = sess.run(gen_loss, feed_dict=feed_dict)
        dis_loss_val = sess.run(dis_loss, feed_dict=feed_dict)
    print('***epoch {0} ***'.format(i))    
    print('gen {0:.4f}, dis {1:.4f}'.format(gen_loss_val, dis_loss_val))
    
    if i%50 == 0:
        sample_gen = sess.run(x_model, feed_dict=feed_dict)
        save_fig(sample_gen, result_dir+'sample_{0}.jpg'.format(i))
#    if (i+1)%100 == 0:
#        saver.save(sess, ckpt_dir+'cell_fig_{}.ckpt'.format(i+1))