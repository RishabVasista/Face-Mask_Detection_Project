import cv2 as cv
import numpy as np
img=cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Photos/pic1.jpg')
cv.imshow('pic2',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('pic',gray)
blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)
#the blur reduces the number of contours
canny=cv.Canny(blur,125,175)
cv.imshow('Canny edges',canny)
contours,heirarchy=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found.')
blank=np.zeros(img.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(0,0,255))
cv.imshow('drawn countours',blank)
cv.waitKey(0)
