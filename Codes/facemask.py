import os
import caer
import numpy as np
import cv2 as cv
import gc
import canaro
import sklearn.model_selection as skm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# Now, we define the size of the image,number of colour channels (grayscale=1)
# and the location of the dataset
IMG_SIZE=(80,80)
channels=1
#fill the path below
char_path=r'C:/Users/Vasista/Desktop/project/data'

# Now we are gonna define a dictionary with the names of the different object
# the number of images for that object and we are going to number it
char_dict= {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))
    # here , we are going through every folder, we are finding the number of images in that folder
# and grabbing the names of the folder and it is stored in char_ditc

#now we sort the dictionary in descending order
char_dict= caer.sort_dict(char_dict,descending=True)
char_dict
# now we can see the dictionary which will be printed

# now we take the elements in a list and store the characters in it.
characters = []
count = 0
for i in char_dict:
   characters.append(i[0])
   count += 1
   if count >=2:
      break
     #here, the count depends on the number of classes used 
characters
# #the names of the characters are printed below

 #now we train the data!
train=caer.preprocess_from_dir(char_path,characters,channels=channels,IMG_SIZE=IMG_SIZE,isShuffle=True)
# # what we're doing here is, it is going to preprocess the images and grab all the images into the training set.
# # a label is created for each class after this.convolution is done at this stage

# #now , we see the length of the training set.
len(train)

# to visualize the images, we'll check the first element
# cv.imshow(train[0][0])
# cv.waitKey(0)

#now, we seperate the feature set and the labels into two seperate lists
featureset,labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

#now we normalize the data in terms of (0,1)
featureset=caer.normalize(featureset)
labels = to_categorical (labels, len(characters))

# Creating train and validation data
split_data = skm.train_test_split(featureset, labels, test_size=.2)
x_train, x_val, y_train, y_val = (np.array(item) for item in split_data)

#we remove unneccesary data to save space below
del train
del featureset
del labels
gc.collect()

# Image data generator (introduces randomness in network ==> better accuracy)
BATCH_SIZE = 32
EPOCHS = 2
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create our model (returns the compiled model)
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9, nesterov=True)

model.summary()

# Training the model

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,steps_per_epoch=len(x_train)//BATCH_SIZE,epochs=EPOCHS,validation_data=(x_val,y_val),validation_steps=len(y_val)//BATCH_SIZE,callbacks = callbacks_list)
print(characters)


# """## Testing"""


img = cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Photos/pic4.jpg')

def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

predictions = model.predict(prepare(img))

# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])
