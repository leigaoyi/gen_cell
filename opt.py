# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:20:34 2019

@author: kasy
"""

from skimage import io
import numpy as np

def save_fig(imgs, fig_name):
    fig = imgs[0, ...]
    fig = np.reshape(fig, [64, 64, 3])
    #fig = 255*fig
    io.imsave(fig_name, fig)
    return 0