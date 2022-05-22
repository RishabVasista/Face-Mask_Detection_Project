import glob
import cv2 as cv

i=0
inputfolder="C:/Users/Vasista/Desktop/graphs"
for img in glob.glob(inputfolder + "/*.png"):
    imag=cv.imread(img)
    hist=cv.resize(imag,(530,410),interpolation=cv.INTER_CUBIC)
    i=i+1
    cv.imwrite("C:/Users/Vasista/Desktop/graphs/resized/Image%04i.png"%i,hist)
