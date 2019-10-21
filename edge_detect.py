# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 04:29:14 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def sobel_vmask(im):
    x = np.array(im,dtype = np.uint8)
    w, h = im.size
    a = np.zeros((h,1),dtype = np.uint8)
    b = np.zeros((1,w+2),dtype = np.uint8)
    y = np.hstack((a,x,a))
    y = np.vstack((b,y,b))
    vmask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    space = np.zeros((h,w),dtype = np.uint8)
    for i in range(h):
        for j in range(w):
            space[i,j] = np.abs(np.sum(y[i:i+3, j:j+3] * vmask)) / 8  
    return space    

def sobel_hmask(im):
    x = np.array(im,dtype = np.uint8)
    w, h = im.size
    a = np.zeros((h,1),dtype = np.uint8)
    b = np.zeros((1,w+2),dtype = np.uint8)
    y = np.hstack((a,x,a))
    y = np.vstack((b,y,b))
    hmask = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    space = np.zeros((h,w),dtype = np.uint8)
    for i in range(h):
        for j in range(w):
            space[i,j] = np.abs(np.sum(y[i:i+3, j:j+3] * hmask)) / 8                   
    return space

imname = input("Image name : ")

try:
    im = Image.open(imname)
    
    if im.mode != 'L':
        im = im.convert('L')
    
    pix1 = sobel_vmask(im)
    pix2 = sobel_hmask(im)
    pix3 = pix1 + pix2

    plt.subplot(1,3,1)
    plt.imshow(pix1)

    plt.subplot(1,3,2)
    plt.imshow(pix2)

    plt.subplot(1,3,3)
    plt.imshow(pix3)
except:
    print("file doesn't exist")
    del imname
    
