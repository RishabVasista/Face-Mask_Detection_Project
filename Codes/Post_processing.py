#Here, we are going to define a post processing function that is used to remove unnecessary hits!
import cv2 as cv
import numpy as np
from collections import deque

facemask_cascade = cv.CascadeClassifier('cascade/cascade.xml')
img = cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Validation/Mask/1701.jpg')
img=cv.resize(img,(int(img.shape[1]*0.2),int(img.shape[0]*0.2)),interpolation=cv.INTER_AREA)
recent_heatmaps = deque(maxlen=10)


def add_heat(heatmap,Facemask_windows):
    """
        Iterate the windows with detected facemasks and enhance the once with highest detections.
    """
    for (x,y,w,h) in Facemask_windows:
        # Add += 1 for all pixels inside each bbox
        heatmap[y:y+h, x:w+x] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    """
        Only keep the detections that have a minimum number of pixels.
    """
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    return heatmap

def Post_process(img,facemask_cascade,Scaling=1.35,neighbour_distance=8,flag=0,min_window_size=[100,100]):
# get the windows where the classifier predicts facemask w/o processing
 #cv.imshow('original',img)
 #blurring the image to reduce noise
 blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
 #cv.imshow('blur',img)
 gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  
 Facemask_windows = facemask_cascade.detectMultiScale(gray, Scaling ,neighbour_distance,flag,min_window_size)
 for (x, y, w, h) in Facemask_windows:
     cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 100), 4)
 #cv.imshow('all detections',img)
 # Thresholding to get better results
 heat = np.zeros_like(img[:,:,0]).astype(np.float)
 heat = add_heat(heat,Facemask_windows)
 #cv.imshow('heatmap of facemasks',heat)
 # Append the detections to detections from last n frames
 recent_heatmaps.append(heat)
    
 # Take the mean of last n frames as discard the windows that are below the threshold
 heatmap = apply_threshold(np.mean(recent_heatmaps, axis=0),1)
 #cv.imshow('after thresholding',heatmap)
 return heatmap

Post_process(img,facemask_cascade,1.1,2,0,[100,100])

cv.waitKey(0)