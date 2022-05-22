#resizing Images
import cv2 as cv
import os
import glob
from PIL import Image

    
inputFolder= 'resized1'
os.mkdir('pre_mask1')
i = 0
for img in glob.glob(inputFolder + "/*.png"):
    imag=cv.imread(img)
    gray=cv.cvtColor(imag,cv.COLOR_BGR2GRAY)
    pro=cv.Canny(gray, 100, 180)
    cv.imwrite("pre_mask1/Image%04i.png"%i,pro)
    i=i+1
print(i)

inputFolder= 'resized2'
os.mkdir('pre_nomask1')
i = 0
for img in glob.glob(inputFolder + "/*.png"):
    imag=cv.imread(img)
    gray=cv.cvtColor(imag,cv.COLOR_BGR2GRAY)
    pro=cv.Canny(gray, 100, 180)
    cv.imwrite("pre_nomask1/Image%04i.png"%i,pro)
    i=i+1
print(i)
