# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:43:13 2019

@author: user
"""

import numpy as np

def step(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)