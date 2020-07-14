#  STEP 1: INSTALLATION & SETUP
#  ===============================================
#  ===============================================
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#  STEP 2: DATA PRE-PROCESSING
#  ===============================================
#  ===============================================

# Importing the dataset 
from PIL import Image
import os, sys
path = "/Users/nhiphamleyen/Desktop/DATASET"

dirs_train_o = os.listdir(path + "/TRAIN/O")
dirs_train_r = os.listdir(path + "/TRAIN/R")
dirs_test_o = os.listdir(path + "/TEST/O")
dirs_test_r = os.listdir(path + "/TEST/R")


# Resize image using Pillow
class_name = ['O', 'R']
# Create training set
x_train = []
x_train.append(dirs_train_o)
x_train.append(dirs_train_r)


# Create testing test
x_test = []
x_test.append(dirs_test_o)
x_test.append(dirs_test_r)


def modify_image(dataset, pathname):
  names = []
  labels = []
  for type in range(2):
    for item in dataset[type]:
      path_tmp = path + "/" + pathname + "/" + class_name[type] + "/"
      if os.path.isfile(path_tmp+item):
        im = Image.open(path_tmp+item)
        imResize = im.resize((32, 32), Image.ANTIALIAS)
        if imResize.mode != 'RGB':
          imResize = imResize.convert('RGB')
        names.append(imResize)
        labels.append(type)
  return names, labels


names_train, labels_train = modify_image(x_train, "TRAIN")
names_test, labels_test = modify_image(x_test, "TEST")


# Train
x_train = np.array([np.array(fname) for fname in names_train])
y_train = np.array(labels_train)


# Test
x_test = np.array([np.array(fname) for fname in names_test])
y_test = np.array(labels_test)


# Normalizing the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0


# Flattening the dataset
x_train.shape, y_train.shape



#  STEP 3: BUILDING THE CNN
#  ===============================================
#  ===============================================

# Defining the object
model = tf.keras.models.Sequential()


# Adding first convolutional layer
# 1) number of filters, filters(kernel) = 32
# 2) kernel size = 3
# 3) padding = same padding or valid padding
# 4) activation = ReLU
# 5) input shape = (32, 32, 3)

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32,32,3]))


# Adding second CNN layer and maxpool layer
# 1) number of filters, filters(kernel) = 32
# 2) kernel size = 3
# 3) padding = same padding or valid padding
# 4) activation = ReLU
# 5) input shape = (32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

# maxpool layer parameters
# 1) pool size = 2
# 2) strides = 2
# 3) padding = valid
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# Adding third CNN layer
# 1) number of filters, filters(kernel) = 64
# 2) kernel size = 3
# 3) padding = same padding or valid padding
# 4) activation = ReLU
# 5) input shape = (32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))


# Adding fourth CNN layer and maxpool layer
# 1) number of filters, filters(kernel) = 64
# 2) kernel size = 3
# 3) padding = same padding or valid padding
# 4) activation = ReLU
# 5) input shape = (32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

# maxpool layer parameters
# 1) pool size = 2
# 2) strides = 2
# 3) padding = valid
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# Adding the dropout layer
model.add(tf.keras.layers.Dropout(0.4))


# Adding the flattening layer
model.add(tf.keras.layers.Flatten())


# Adding first dense layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))


# Adding second dense layer (output layer)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#  STEP 4: COMPILE THE MODEL

#  Compiling the model
#  ===============================================
#  ===============================================

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])


model.fit(x_train, y_train, batch_size=10, epochs=10)





