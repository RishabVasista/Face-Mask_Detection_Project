import pickle
import cv2 as cv
import math
import glob
import time
import cv2 as cv
from collections import deque
import numpy as np
import matplotlib.image as mpimg

import pickle

from skimage.feature import hog
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from svm_dtree import setup_train_data

colorConv = 'BGR2HSV'
hog_channel = "ALL"
orient = 9
pix_per_cell = 8
cell_per_block = 2
recent_heatmaps = deque(maxlen=10)
filename = 'facemask_model.sav'
X_train, X_test, y_train, y_test, X_scaler = setup_train_data(colorConv, orient, pix_per_cell, cell_per_block, hog_channel)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
