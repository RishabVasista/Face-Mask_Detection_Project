import cv2 as cv
import glob
i=1
inputFolder= 'C:/Users/Vasista/Desktop/101FARE_'
for img in glob.glob(inputFolder + "/*.jpg"):
    imag=cv.imread(img)
    b,g,r = cv.split(imag)
    histb=cv.equalizeHist(b)
    histg=cv.equalizeHist(g)
    histr=cv.equalizeHist(r)
    hist=cv.merge([histb,histg,histr])
    cv.imwrite("C:/Users/Vasista/Desktop/processed/farewell%04i.jpg"%i,hist)
    i=i+1