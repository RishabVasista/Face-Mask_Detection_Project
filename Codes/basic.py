import cv2 as cv
img=cv.imread('Image0067.png')
cv.imshow('pic2',img)

#converting to greyscale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('greypic',gray)

#Blur
blur=cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
#kernel size must be an odd value
cv.imshow('blurpic',blur)

#canny cascade (edge detection)
canny=cv.Canny(img,125,175)
cv.imshow('Canny edges',canny)

#dialation
dialated=cv.dilate(canny,(3,3),iterations=3)
cv.imshow('Dialation',dialated)

#erosion
eroded=cv.erode(dialated,(3,3),iterations=3)
cv.imshow('Eroded',eroded)


cv.waitKey(0)
