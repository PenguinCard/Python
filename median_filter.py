# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:34:39 2019

@author: KICT-02
"""

import numpy as np
from PIL import Image

def median_filter(pix, m_size):
    im = Image.fromarray(pix,'L')
    col, row = im.size
    a = pix[row-int(m_size/2):,col-int(m_size/2):]
    b = pix[row-int(m_size/2):,:]
    c = pix[row-int(m_size/2):,:int(m_size/2)]
    d = pix[:,col-int(m_size/2):]
    e = pix[:,:int(m_size/2)]
    f = pix[:int(m_size/2),col-int(m_size/2):]
    g = pix[:int(m_size/2),:]
    h = pix[:int(m_size/2),:int(m_size/2)]
    part1 = np.hstack([a, b, c])
    part2 = np.hstack([d, pix, e])
    part3 = np.hstack([f, g, h])
    pix_all = np.vstack([part1, part2, part3])
    
    space = np.zeros((row,col),dtype = np.uint8)
    
    for i in range(row):
        for j in range(col):
            part = pix_all[i:i+m_size,j:j+m_size]
            re_part = part.reshape(m_size**2)
            re_part.sort()
            space[i, j] = re_part[int((m_size**2)/2)]
    return space

imname = input('Image name : ')
f_size = input('filter size : ')

m_size = np.uint8(f_size)

im = Image.open(imname)
pix = np.array(im, dtype=np.uint8)

pix1 = median_filter(pix, m_size)

im1 = Image.fromarray(pix1,'L')
im1.show()

# using uint8_image / circular padding
        