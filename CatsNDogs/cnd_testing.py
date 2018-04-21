import cv2
import os
import sys
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove tensorflow verbose 

modelLocation = sys.argv[1] # Baseline.h5

testImgLocation = '10.png' #cat [1. 0.]
testImgClass = [0]

print 'Loading model...'
model = load_model(modelLocation)


testImg = cv2.resize(cv2.imread(testImgLocation), (224,224))
testImg = np.array([testImg])

testImg = np.reshape(testImg, (1, 224, 224, 3))


print 'Loading Image...'
testImg = np.array([cv2.imread(testImgLocation)])
testImgClass = np.array([testImgClass])
testImgClass = np_utils.to_categorical(testImgClass,2) #since we have 2 classes [cat, dog]

print 'Accuracy', model.evaluate(testImg, testImgClass)[1]