import cv2 as cv
import numpy as np
from collections import deque
from scipy.ndimage.measurements import label


# Load the cascade
face_cascade = cv.CascadeClassifier('cascade/cascade.xml')
recent_heatmaps = deque(maxlen=10)

# To capture video from webcam. 
cap = cv.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

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

def draw_labeled_bboxes(img, labels):
    """
        Draw the boxes on the detected masks
    """
    for i in range(1, labels[1]+1):
        # Find pixels with each mask label value
        nonzero = (labels[0] == i).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv.rectangle(img, bbox[0], bbox[1], (255,255,255), 2)
    return img

def Post_process(img,facemask_cascade,Scaling=1.35,neighbour_distance=8,flag=0,min_window_size=[100,100]):
# get the windows where the classifier predicts facemask w/o processing
 #blurring the image to reduce noise
 blur = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
 gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  
 Facemask_windows = facemask_cascade.detectMultiScale(gray, Scaling ,neighbour_distance,flag,min_window_size)
 # Thresholding to get better results
 heat = np.zeros_like(img[:,:,0]).astype(np.float)
 heat = add_heat(heat,Facemask_windows)
 # Append the detections to detections from last n frames
 recent_heatmaps.append(heat)
    
 # Take the mean of last n frames as discard the windows that are below the threshold
 heatmap = apply_threshold(np.mean(recent_heatmaps, axis=0),1.2)
  # Add labels to remaning detections
 labels = label(heatmap)
 # Draw boxes on the cars and return the image
 return draw_labeled_bboxes(img, labels)

while True:
    # Read the frame
    _, img = cap.read()

    img = Post_process(img,face_cascade,1.3,4)
    
    # Display
    cv.imshow('processed detection', img)

    # Stop if escape key is pressed
    k = cv.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
cv.destroyAllWindows()
