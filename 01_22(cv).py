# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 01:00:30 2020

@author: user
"""

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret == False:
        break

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0:
        break
   
cap.release()
cv2.destroyAllWindows()
