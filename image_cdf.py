# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 03:26:02 2019

@author: user
"""

import numpy as np
from PIL import Image

def cdf(x, w, h): 
    y = x.cumsum() / (w * h)
    return y

im = Image.open('Fig0308(a)(pollen).tif')
pix = np.array(im)
height, width = im.size
hist,bin = np.histogram(pix,256,[0,256])
cdf1 = cdf(hist, width, height)
