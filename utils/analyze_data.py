# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:03:42 2018

@author: CX
"""
import os
import numpy as np

data_dir = os.path.dirname(os.path.dirname(__file__)) + "/baseline/data/"

# class ratio
labels = np.loadtxt(data_dir+"labels.txt", dtype = np.int32)
print(len(labels))
print(np.sum(labels)/len(labels))