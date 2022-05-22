import cv2 as cv
import numpy as np

blank= np.zeros((500,500,3),dtype='uint8')
cv.imshow('blank',blank)

blank[:]=0,255,0
#it uses rgb colours
cv.imshow('green',blank)

blank[200:300,300:400]=0,0,255
cv.imshow('red box',blank)

cv.rectangle(blank,(0,0),(250,250),(166,255,0),2)
#instead of thickness of 2 we can give it as cv.FILLED or -1 to fill the colour. 
cv.imshow('rectangle',blank)

#drawing a line
cv.line(blank,(0,0),(250,250),(166,255,90),2)
#instead of thickness of 2 we can give it as cv.FILLED or -1 to fill the colour. 
cv.imshow('line',blank)

cv.circle(blank,(250,250),60,(166,255,100),2)
#instead of thickness of 2 we can give it as cv.FILLED or -1 to fill the colour. 
cv.imshow('circle',blank)

# adding text
cv.putText(blank,'hey',(233,220),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,233),2)
cv.imshow('Text',blank)
cv.waitKey(0)