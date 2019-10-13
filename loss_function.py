# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:01:39 2019

@author: user
"""

import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
