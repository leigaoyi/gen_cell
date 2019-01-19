# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:20:34 2019

@author: kasy
"""

from skimage import io
import numpy as np

def save_fig(imgs, fig_name):
    fig1 = imgs[0, ...]
    fig2 = imgs[1, ...]
    fig3 = imgs[2, ...]
    fig4 = imgs[3, ...]
    fig_row_1 = np.concatenate([fig1, fig2], axis=1)
    fig_row_2 = np.concatenate([fig3, fig4], axis=1)
    fig = np.concatenate([fig_row_1, fig_row_2], axis=0)
    #fig = 255*fig
    io.imsave(fig_name, fig)
    return 0