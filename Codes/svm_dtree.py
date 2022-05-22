import os
import math
import glob
import time
import cv2 as cv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.feature import hog
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from scipy.ndimage.measurements import label

import pickle

# Hold the color code name and opencv objects in a dict for easy conversion
colorCodeDict = {
    'RGB2GRAY' : cv.COLOR_RGB2GRAY,
    'RGB2RGBA' : cv.COLOR_RGB2RGBA,
    'RGB2BGR' : cv.COLOR_RGB2BGR,
    'RGB2BGRA' : cv.COLOR_RGB2BGRA,
    'RGB2HSV' : cv.COLOR_RGB2HSV,
    'RGB2HLS' : cv.COLOR_RGB2HLS,
    'RGB2LUV' : cv.COLOR_RGB2LUV,
    'RGB2YUV' : cv.COLOR_RGB2YUV,
    'RGB2YCrCb' : cv.COLOR_RGB2YCrCb,
    'BGR2GRAY' : cv.COLOR_BGR2GRAY,
    'BGR2BGRA' : cv.COLOR_BGR2BGRA,
    'BGR2RGB' : cv.COLOR_BGR2RGB,
    'BGR2RGBA' : cv.COLOR_BGR2RGBA,
    'BGR2HSV' : cv.COLOR_BGR2HSV,
    'BGR2HLS' : cv.COLOR_BGR2HLS,
    'BGR2LUV' : cv.COLOR_RGB2LUV,
    'BGR2YUV' : cv.COLOR_RGB2YUV,
    'BGR2YCrCb' : cv.COLOR_RGB2YCrCb
}

# define color conversion function

def convert_color(img , convCode='BGR2GRAY'):
    """
        return image converted to required colospace
    """
    return cv.cvtColor(img, colorCodeDict[convCode])

# functions used to extract different image features
def bin_spatial(img, size=(32, 32)):
    """
        Return the image color bins
    """
    color1 = cv.resize(img[:,:,0], size).ravel()
    color2 = cv.resize(img[:,:,1], size).ravel()
    color3 = cv.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):
    #   Return all channel histogram.
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

def get_hog_features(img, orient, pix_per_cell, cell_per_block,feature_vector=True):
    """
        Return a histogram of oriented gradients using skimage.
    """
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualize=False, feature_vector=feature_vector)

#now. we define a function to extract and combine different features
def extract_features(imgs, colorConv, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
        Wrapper function to return a bag of features by combining different features extracted with above functions.
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images and extract features
    for file in imgs:
        image = cv.imread(file)
        feature_image = convert_color(image, colorConv)
        spatial_features = bin_spatial(feature_image)
        hist_features = color_hist(feature_image)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    return features

#Function to train different classifiers
def train_SVC(X_train, y_train):
    """
        Function to train an svm.
    """
    svc = svm.LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc

def train_dtree(X_train, y_train):
    """
        Function to train a decision tree.
    """
    clf = tree.DecisionTreeClassifier()
    t=time.time()
    clf = clf.fit(X_train, y_train)
    t2=time.time()
    print(round(t2-t, 2), 'Seconds to train dtree...')
    return clf
'''
def test_classifier(svc, X_test, y_test):
    """
        Funtion to test the classifier.
    """
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    pred = svc.predict(X_test[0:n_predict])
    actual = y_test[0:n_predict]
    print('My SVC predicts: ', pred)
    print('For these',n_predict, 'labels: ', actual)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')'''
def test_classifier(svc, X_test, y_test):
    """
        Funtion to test the classifier.
    """
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    pred = svc.predict(X_test[0:n_predict])
    actual = y_test[0:n_predict]
    print('My SVC predicts: ', pred)
    print('For these',n_predict, 'labels: ', actual)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# function to remove FP and duplicates
def add_heat(heatmap, bbox_list):
    """
        Iterate the windows with detected masks and enhance the once with highest detections.
    """
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
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
        # Find pixels with each car label value
        nonzero = (labels[0] == i).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

# function to detect masks
def find_masks(img, colorConv, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    """
        This function takes in an image, extracts the features from a region of interest and
        runs the predictions on the features.
        Returns a list of co-ordinates where mask is detected.
    """
    img = img.astype(np.float32)/255
    img_shape = img.shape
    # Crop the image
    ystart = math.floor(img_shape[0]*.55)
    ystop = math.floor(img_shape[0]*.85)
    img = img[ystart:ystop,:,:]
    #plot_img(img_tosearch, True)
    img = convert_color(img, colorConv)
    # Define blocks and steps as above
    nxblocks = (img.shape[1] // pix_per_cell)-1
    nyblocks = (img.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # set the window size same as the test image size
    window = 100
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog_ch1 = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, feature_vector=False)
    hog_ch2 = get_hog_features(img[:,:,1], orient, pix_per_cell, cell_per_block, feature_vector=False)
    hog_ch3 = get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, feature_vector=False)
    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog_ch1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog_ch2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog_ch3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = cv.resize(img[ytop:ytop+window, xleft:xleft+window], (64,64))
            # Get color features
            spatial_features = bin_spatial(subimg)
            # Get histogram feature
            hist_features = color_hist(subimg)
            # add all features and Scale them
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            # make a prediction
            test_prediction = svc.predict(test_features)
            # Add to list of windows if car predicted
            if test_prediction == 1:
                xbox_left = np.int(xleft)
                ytop_draw = np.int(ytop)
                win_draw = np.int(window)
                on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return on_windows

# parameters
colorConv = 'BGR2HSV'
hog_channel = "ALL"
orient = 9
pix_per_cell = 8
cell_per_block = 2
recent_heatmaps = deque(maxlen=10)

# setting up training and validation data
def setup_train_data(colorConv, orient, pix_per_cell, cell_per_block, hog_channel):
    """
        Setup data for classifier training. 
        Shuffle the data and split it in training and testing set.
    """
    masks = []
    images = glob.glob('C:/Users/Vasista/Desktop/project/openCV tutorial files/resized11/**/*.png', recursive=True)
    for image in images:
        masks.append(image)
    
    images = glob.glob('C:/Users/Vasista/Desktop/project/openCV tutorial files/resized22/**/*.png', recursive=True)
    notmasks = []
    for image in images:
        notmasks.append(image)

    mask_features = extract_features(masks, colorConv, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    notmask_features = extract_features(notmasks, colorConv, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    # Create an array stack of feature vectors
    X = np.vstack((mask_features, notmask_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y_train = np.hstack((np.ones(len(mask_features)), np.zeros(len(notmask_features))))
    # shuffle the data
    #X_train, y_train = shuffle(scaled_X, y_train)
    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y_train, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test, X_scaler

print('Preparing training data...')
X_train, X_test, y_train, y_test, X_scaler = setup_train_data(colorConv, orient, pix_per_cell, cell_per_block, hog_channel)
print("Number of training examples =", len(X_train))
print("Number of testing examples =", len(X_test))

# Training the classifier
print('Training Classifier...')
svc = train_SVC(X_train, y_train)
#clf = train_dtree(X_train, y_train)

#saving the model
filename = 'facemask_model_svm.sav'
#filename = 'facemask_model.sav'
pickle.dump(svc, open(filename, 'wb'))
filename_XScaler= 'XScaler_model_svm.sav'
pickle.dump(X_scaler, open(filename_XScaler, 'wb'))

