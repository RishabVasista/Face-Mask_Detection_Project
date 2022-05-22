#reading a video
import numpy as np
import cv2 as cv
cap=cv.VideoCapture(0)
while True:

    ret, frame=cap.read()
    
    if ret==True:
        cv.imshow('your name',frame)
   
    if cv.waitKey(30) & 0xFF==ord('d'):
         break            
    

cap.release()
cv.destroyAllWindows()