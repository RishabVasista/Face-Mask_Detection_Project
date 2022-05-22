#reading a video
import numpy as np
import cv2 as cv
cap=cv.VideoCapture(0)
face_cascade=cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_frontalface_default.xml')
while True:

    ret, frame=cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.1, 3)
    for (x, y, w, h) in faces:
      cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv.imshow('your name',frame)
    if cv.waitKey(30) & 0xFF==ord('d'):
         break            
    

cap.release()
cv.destroyAllWindows()