import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Photos/pic1.jpg')
cv.imshow('Pic', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)
#split images are in grayscale form
cv.imshow('Blue comp', b)
cv.imshow('Green comp', g)
cv.imshow('Red comp', r)
# to show them as their respective colours, we use the following code.
blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

#voila!
cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)