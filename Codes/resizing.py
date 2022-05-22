#resizing Images
import cv2 as cv
import os
import glob
from PIL import Image

    
inputFolder= 'positive1'
os.mkdir('resized1')
i = 0
for img in glob.glob(inputFolder + "/*.png"):
    imag=cv.imread(img)
    h,w = imag.shape[:2]
    if h<100 // w<100 :
     imgResized=cv.resize(imag,(100,100),interpolation=cv.INTER_CUBIC)
     gray=cv.cvtColor(imgResized,cv.COLOR_BGR2GRAY)
     hist=cv.equalizeHist(gray)
    else :
     imgResized=cv.resize(imag,(100,100),interpolation=cv.INTER_AREA)
     gray=cv.cvtColor(imgResized,cv.COLOR_BGR2GRAY)
     hist=cv.equalizeHist(gray)
    cv.imwrite("resized1/Image%04i.png"%i,hist)
    i=i+1
print(i)

inputFolder= 'negative1'
os.mkdir('resized2')
i = 0
for img in glob.glob(inputFolder + "/*.png"):
    imag=cv.imread(img)
    imgResized=cv.resize(imag,(100,100),interpolation=cv.INTER_AREA)
    cv.imwrite("resized2/Image%04i.png"%i,imgResized)
    i=i+1
print(i)
