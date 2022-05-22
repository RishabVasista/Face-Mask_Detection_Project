import cv2 as cv
import os
import glob
from skimage import feature
import numpy as np
from Post_processing import Post_process

#loading image folder and classifier
PositiveFolder= 'Validation/Mask_R'
NegativeFolder= 'Validation/non_mask_R'
face_mask_cascade=cv.CascadeClassifier('test7/cascade.xml')
#checking for True Positive and True negative

for img in glob.glob(PositiveFolder + "/*.jpg"):
    image = cv.imread(img)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    masks = face_mask_cascade.detectMultiScale(gray, 1.05, 4,0,[100,100])
    for (x,y,w,h) in masks :
        cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=1)
    cv.imshow('mask',image)
    k = cv.waitKey(0) & 0xff
    cv.destroyWindow('mask')

for img in glob.glob(PositiveFolder + "/*.png"):
    image = cv.imread(img)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    masks = face_mask_cascade.detectMultiScale(gray, 1.05, 4,0,[100,100])
    for (x,y,w,h) in masks :
        cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=1)
    cv.imshow('mask',image)
    k = cv.waitKey(0) & 0xff
    cv.destroyWindow('mask')
#checking for False Positive and False Negative
print("non mask pics")
cv.waitKey(10000)
for img in glob.glob(NegativeFolder + "/*.jpg"):
    image = cv.imread(img)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    masks = face_mask_cascade.detectMultiScale(gray, 1.05, 4,0,[100,100])
    for (x,y,w,h) in masks :
        cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=1)
    cv.imshow('no Mask',image)
    k = cv.waitKey(0) & 0xff
    cv.destroyWindow('no Mask')

