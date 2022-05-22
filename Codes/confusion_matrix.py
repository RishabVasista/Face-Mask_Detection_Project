import cv2 as cv
import os
import glob
from skimage import feature
import numpy as np

#loading image folder and classifier
PositiveFolder= 'Validation/Mask'
NegativeFolder= 'Validation/Non Mask'
face_mask_cascade=cv.CascadeClassifier('cascade/cascade.xml')
#face_mask_cascade=cv.CascadeClassifier('test5/cascade.xml')
#checking for True Positive and False Negative
TP=0
FN=0
i=0
for img in glob.glob(PositiveFolder + "/*.png"):
    image = cv.imread(img)
    masks = face_mask_cascade.detectMultiScale(image, 1.05, 3)
    if len(masks)==1 :
        TP=TP+1
    else:
        FN=FN+1
    if i<=450 :
        i=i+1 
    else :
        break       
#checking for False Positive and True Negative
TN=0
FP=0
i=0
for img in glob.glob(NegativeFolder + "/*.png"):
    image = cv.imread(img)
    masks = face_mask_cascade.detectMultiScale(image, 1.05, 3)
    if len(masks)==0:
        TN=TN+1
    else:
        FP=FP+1
    if i<=450 :
        i=i+1 
    else :
        break 

print("True Positives:",TP)
print("False Negatives:",FN)
print("True Negatives:",TN)
print("False Positives:",FP)