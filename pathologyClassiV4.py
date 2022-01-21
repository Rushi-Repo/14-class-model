import keras
from keras import backend as K
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import cv2
import sklearn
import shap
import os
import seaborn as sns
import time
import pickle


from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('/home/narayana/wd/codes/opacitySegmentation/xrayClass14/testImages/'):
    f.extend(filenames)
    print(filenames)
    
    break

classLabels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential

def load_C3M3_model():
   
    # create the base pre-trained model
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='/home/narayana/wd/codes/opacitySegmentation/xrayClass14/densenet.hdf5')
    print("Loaded DenseNet")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions = Dense(len(classLabels), activation="sigmoid")(x)
    print("Added layers")

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print("Compiled Model")

    #model.load_weights("/home/narayana/wd/codes/opacitySegmentation/xrayClass14/densenet.hdf5")
    print("Loaded Weights")
    return model

model = load_C3M3_model()

img_size = 224
img = cv2.imread("/home/narayana/wd/codes/opacitySegmentation/xrayClass14/testImages/01.PNG") 
img = cv2.resize(img, (img_size, img_size))
img = np.reshape(img, [1, img_size, img_size, 3])
imgOut = model.predict(img)
classLabels[np.argmax(imgOut)]
pd.DataFrame(np.round(imgOut.tolist()[0],3), classLabels, columns=['probability'])


for fName in filenames:
    fileName = '/home/narayana/wd/codes/opacitySegmentation/xrayClass14/testImages/' + fName
    print(fName)
    img = cv2.imread(fileName) 
    img = cv2.resize(img, (img_size, img_size))
    img = np.reshape(img, [1, img_size, img_size, 3])
    imgOut = model.predict(img)
    classLabels[np.argmax(imgOut)]
    pathPredictions = pd.DataFrame(np.round(imgOut.tolist()[0],3), classLabels, columns=['probability'])
    print(pathPredictions)
