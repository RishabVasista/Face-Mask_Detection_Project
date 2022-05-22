import cv2 as cv

img=cv.imread('C:/Users/Vasista/Desktop/wo_median_blur.jpg')
blurred=cv.medianBlur(img,9)
cv.imwrite("C:/Users/Vasista/Desktop/w_median_blur.jpg",blurred)