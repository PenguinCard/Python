# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 03:26:02 2019

@author: user
"""

import numpy as np
from PIL import Image

def histscale(x): # histogram scaling(linear.ver)
    Max = np.max(x)
    Min = np.min(x)
    a = np.uint8(255/(Max - Min)*(x - Min))
    return a

def cdf(x, w, h):  # image CDF x = image / w = width / h = height
    y = x.cumsum() / (w * h)
    return y

def histequal(x, w, h): # histogram equalization x = image / w = width / h = height
    t = np.zeros((h,w),dtype = np.uint8)
    hist,bin = np.histogram(x,256,[0,256])
    imcdf = np.uint8(cdf(hist, w, h) * 255)

    for i in range(h): # repeat row
        for j in range(w): # repeat col
            t[i,j] = imcdf[pix[i,j]]
    return t

im = Image.open('Fig0308(a)(pollen).tif')
im.show()
pix = np.array(im,dtype = np.uint8)
width, height = im.size # image width, height

im1 = Image.fromarray(histscale(pix))
im1.show()

im2 = Image.fromarray(histequal(pix, width, height),'L')

im2.show()

# All processes are using uint8 data type
