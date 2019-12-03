# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:44:07 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

def MSE(y, t):
    return 0.5 * np.sum((y-t)**2)

class onelayernet:
    def __init__(self, x, t):
        self.x = x
        self.t = t
        self.W = np.ones((1,1),dtype=np.float64)
        self.b = np.random.randn(1,1)
        self.lr = 1e-4

    def predict(self):
        y = np.dot(self.x,self.W)+self.b
        return y
    
    def loss(self):
        return MSE(self.predict(), self.t)
            
    def gradient_W(self):
        return 0.02*np.dot(self.x.T,(self.predict() - self.t))
    
    def gradient_b(self):
        return 0.02*(self.b - self.loss())
    
    def train(self):
        for epoch in range(100):
            self.W -= self.lr * self.gradient_W()
            self.b -= self.lr * self.gradient_b()
            if epoch % 10 == 0:
                print('epoch:', epoch, 'loss:', self.loss())
                
datasets = np.loadtxt('data.txt')
x_data = datasets[:,0].reshape(100,1)
t_data = datasets[:,1].reshape(100,1)
   
OLN = onelayernet(x_data, t_data)
OLN.train()

plt.plot(x_data, OLN.predict(), 'r')
plt.scatter(x_data, t_data)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['prediction', 'actual scatter'])
plt.show()