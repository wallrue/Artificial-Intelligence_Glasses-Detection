# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:24:34 2022

@author: mylocalaccount
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import shutil
import cv2

import tensorflow as tf
import keras
import tensorflow.keras as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Input, Add,AveragePooling2D
from tensorflow.keras.models import Model
from keras.initializers import glorot_uniform

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


args={'dataset':'dataset','model':'my.model','plot':'plot.png'}

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

path_dataset = "images"
model_type = "LeNet"

class LeNet:

    def __init__(self, input_shape = (224, 224, 3), classes = 2): 
        
        # initialize the model
        model = Sequential()
        inputShape = input_shape #(height, width, depth)
        
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Create model
        self.model = model
    
    def get_model(self):
        return self.model

class ResNet50:
    
    def convolutional_block(self, X, f, filters, stage, block, s = 2):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###

        # Second component of main path (≈3 lines)
        X = Conv2D(F2, (f,f), strides = (1,1), padding='same',name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)
        
        return X

    def identity_block(self, X, f, filters, stage, block):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###
        
        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    
    def __init__(self, input_shape = (224, 224, 3), classes = 2):
 
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape) #.reshape(-1,32,32,1)
    
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
        # Stage 2
        X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
        # Stage 3 
        X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
        # Stage 4 
        X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
        # Stage 5 
        X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
        # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D()(X)
    
        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        
        # Create model
        self.model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    def get_model(self):
        return self.model
    

if __name__ == "__main__":
    
      
    """--
    PREPARE DATASET 
    ---"""
    
    data = []
    labels = []  
    
    # handle with the dataset path
    imagePaths = sorted(list(paths.list_images(path_dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    
    for imagePath in tqdm(imagePaths, desc="Loading dataset"):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (100, 100))
        image = img_to_array(image)
        data.append(image)
    
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
    
        # class info is based on the folder named ‘pos’ 
        label = 1 if label == "glasses" else 0
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)
    
    # construct the image generator for data augmentation – adding additional images 
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.2, 
                              zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

    """--
    TRAINING MODEL
    ---"""
    
    # initialize the model
    model = LeNet(input_shape = (100, 100, 3), classes = 2)
    #model = ResNet50(input_shape = (100, 100, 3), classes = 2)
    mymodel = model.get_model()
 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=INIT_LR, decay_steps=EPOCHS, decay_rate=INIT_LR / EPOCHS)
    opt = Adam(learning_rate=lr_schedule) 
    mymodel.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["acc"])
    
    # train the network
    H = mymodel.fit(aug.flow(trainX, trainY, batch_size=BS),
            validation_data=(testX, testY), steps_per_epoch=int(len(trainX)/BS),
            epochs=EPOCHS, verbose=1)
    mymodel.save(args["model"])
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=[14,12])
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on POS/NEG")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(args["plot"])
