import cv2 as cv
import time
facemask_cascade=cv.CascadeClassifier('cascade\cascade.xml')
def detect(img):
    #facemask_cascade=cv.CascadeClassifier('cascade\cascade.xml')    
    t=time.time()
    facemask=facemask_cascade.detectMultiScale(img,1.2,3)
    t2=time.time()
    print(round(t2-t, 3), 'Seconds to detect...')
    return facemask

test=cv.imread('Image0067.png')
fm= detect (test)