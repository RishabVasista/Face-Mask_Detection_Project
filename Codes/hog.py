from skimage import feature
import cv2
import matplotlib.pyplot as plt
import os
import glob

inputFolder='resized1'
os.mkdir('preprocess_mask')
i=0
for img in glob.glob(inputFolder + "/*.png"):
 image = cv2.imread('resized1/Image1290.png')
 blur = cv2.GaussianBlur(image,(3,3), 0)
 (hog, hog_image) = feature.hog(blur, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, transform_sqrt=True)
 #cv2.imshow('HOG Image', hog_image)
 #cv2.waitKey(30)
 print (i)
 cv2.imwrite("preprocess_mask/Image%04i.png"%i,hog_image*255.)
 i=i+1