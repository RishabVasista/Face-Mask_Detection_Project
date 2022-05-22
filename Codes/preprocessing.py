import glob
import cv2 as cv
from skimage import feature
from scipy.ndimage.measurements import label
import os
import matplotlib.pyplot as plt


# parameters
colorConv = cv.COLOR_BGR2GRAY
hog_channel = "ALL"
orient = 9
pix_per_cell = 8
cell_per_block = 2
inputFolder = 'resized1'

os.mkdir('preprocess_mask')
i=0
for img in glob.glob(inputFolder + "/*.png"):
 (hog, hog_image) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, transform_sqrt=True)
 cv.imwrite("preprocess_mask/Image%04i.png"%i,hog_image*255.)
 i=i+1
for img in glob.glob(inputFolder + "/*.jpg"):
 (hog, hog_image) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, transform_sqrt=True)
 cv.imwrite("preprocess_mask/Image%04i.jpg"%i,hog_image*255.)
 i=i+1
print(i)
inputFolder = 'resized2'
i=0
os.mkdir('preprocess_nomask')
for img in glob.glob(inputFolder + "/*.png"):
 (hog, hog_image) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, transform_sqrt=True)
 cv.imwrite("preprocess_nomask/Image%04i.png"%i,*255.)
 i=i+1
print(i)