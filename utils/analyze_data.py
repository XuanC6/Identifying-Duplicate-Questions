# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:03:42 2018

@author: CX
"""
import os
import numpy as np

data_dir_1 = os.path.dirname(os.path.dirname(__file__)) + "/DupQues/data/train/"
data_dir_2 = os.path.dirname(os.path.dirname(__file__)) + "/DupQues/data/valid/"
data_dir_3 = os.path.dirname(os.path.dirname(__file__)) + "/DupQues/data/test/"

# class ratio
labels_1 = np.loadtxt(data_dir_1+"labels.txt", dtype = np.int32)
print(len(labels_1))
print(np.sum(labels_1)/len(labels_1))

labels_2 = np.loadtxt(data_dir_2+"labels.txt", dtype = np.int32)
print(len(labels_2))
print(np.sum(labels_2)/len(labels_2))

labels_3 = np.loadtxt(data_dir_3+"labels.txt", dtype = np.int32)
print(len(labels_3))
print(np.sum(labels_3)/len(labels_3))