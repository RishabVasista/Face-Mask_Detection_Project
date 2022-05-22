import cv2 as cv
import numpy as np

img=cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Photos/pic2.jpg')
cv.imshow('bike',img)
#Translation
def translate (img,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

translated=translate(img,100,100)
cv.imshow('translated pic',translated)

#Rotation
def Rotate (img,angle,rotpoint=None):
    (height,width) = img.shape[:2]

    if rotpoint is None:
        rotpoint=(width//2 ,height//2)

    rotMat=cv.getRotationMatrix2D(rotpoint,angle,1.0)
    dimensions=(width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated=Rotate(img,45)
cv.imshow('Rotated pic',rotated)


#Resizing
resized=cv.resize(img,(500,500),interpolation=cv.INTER_LINEAR)
cv.imshow('Resized image',resized)

#Flipping images
flip=cv.flip(img,0)
cv.imshow('Fliped image',flip)
# 1  ---> horizontal flip
# 0  ---> vertical flip
# -1 ---> both horizontal and vertical flip

#cropping
cropped=img[200:400,300:400]
#here , image slicing is used.
cv.imshow('Cropped image',cropped)

cv.waitKey(0)