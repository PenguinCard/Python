# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.75
    tmp = np.sum(w*x) + b
    if tmp < 0:
        return 0
    else:
        return 1
        
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.75
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
        
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.25
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        if (x1 < 0.5) and (x2 < 0.5):
            return 0
        else:
            return 1
        
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
