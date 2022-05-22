#resizing Images
import cv2 as cv
import os
import glob
from PIL import Image

    
inputFolder= 'Validation/Mask'
os.mkdir('Validation/Mask_R')
i = 0
for img in glob.glob(inputFolder + "/*.jpg"):
    imag=cv.imread(img)
    h,w = imag.shape[:2]
    if h<50 // w<50 :
     imgResized=cv.resize(imag,(500,500),interpolation=cv.INTER_CUBIC)
    else :
     imgResized=cv.resize(imag,(500,500),interpolation=cv.INTER_AREA)
    cv.imwrite("Validation/Mask_R/Image%04i.jpg"%i,imgResized)
    i=i+1
for img in glob.glob(inputFolder + "/*.png"):
    imag=cv.imread(img)
    h,w = imag.shape[:2]
    if h<50 // w<50 :
     imgResized=cv.resize(imag,(500,500),interpolation=cv.INTER_CUBIC)
    else :
     imgResized=cv.resize(imag,(500,500),interpolation=cv.INTER_AREA)
    cv.imwrite("Validation/Mask_R/Image%04i.png"%i,imgResized)
    i=i+1    
print(i)

inputFolder= 'Validation/Non Mask'
os.mkdir('Validation/non_mask_R')
i = 0
for img in glob.glob(inputFolder + "/*.jpg"):
    imag=cv.imread(img)
    imgResized=cv.resize(imag,(500,500),interpolation=cv.INTER_AREA)
    cv.imwrite("Validation/non_mask_R/Image%04i.jpg"%i,imgResized)
    i=i+1
print(i)
