# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 03:26:02 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def histscale(pix): # histogram scaling(linear.ver)
    x = np.array(pix,dtype = np.uint8)
    Max = np.max(x)
    Min = np.min(x)
    a = np.uint8(255/(Max - Min)*(x - Min))
    return a

def cdf(x, w, h):  # image CDF x = image / w = width / h = height
    y = x.cumsum() / (w * h)
    return y

def histequal(pix): # histogram equalization x = image / w = width / h = height
    x = np.array(pix,dtype = np.uint8)
    w, h = pix.size
    t = np.zeros((h,w),dtype = np.uint8)
    hist,bin = np.histogram(x,256,[0,256])
    imcdf = np.uint8(cdf(hist, w, h) * 255)

    for i in range(h): # repeat row
        for j in range(w): # repeat col
            t[i,j] = imcdf[x[i,j]]
    return t

imname = input("Image name : ")

try:
    im = Image.open(imname)
    if im.mode != 'L':
        im = im.convert('L')
    plt.subplot(1,3,1)
    plt.imshow(im)

    im1 = Image.fromarray(histscale(im))
    plt.subplot(1,3,2)
    plt.imshow(im1)

    im2 = Image.fromarray(histequal(im),'L')

    plt.subplot(1,3,3)
    plt.imshow(im2)
except:
    print("file doesn't exist")
    del imname

# All processes are using uint8 data type
