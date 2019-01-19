# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:01:42 2019

@author: kasy
"""

import glob
from skimage import io

data_dir = './data/train/2/*.jpg'
data_name_list = glob.glob(data_dir)
data_list = []
for i in data_name_list: #[:1]:
    data_list.append(io.imread(i))
process_data = []
for i in data_list: # 255-->(0,1)
    fig = i/255.*2 - 1
    process_data.append(fig)   
    
    