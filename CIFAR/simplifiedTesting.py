import sys
import os
from keras.models import model_from_json
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.utils import np_utils

testImgLocation = sys.argv[1] # testImg.png
testImgClass = sys.argv[2] # 8

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove tensorflow verbose 



print 'Loading Classification Model...'
json_file = open('cifar10classification_top5.json', 'r') # load json and create model
classif_model_json = json_file.read()
json_file.close()
classif_model = model_from_json(classif_model_json)
classif_model.load_weights('cifar10classification_top5.h5') # load weights into new model
print '\tLoaded model from disk'
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classif_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print  '\tCompiled model'

print 'Loading Image...'
testImg = np.array([cv2.imread(testImgLocation)])
testImgClass = np.array([testImgClass])
print '\tConverting image'
testImg = testImg.astype('float32')/255.
testImg = np.reshape(testImg, (len(testImg), 32, 32, 3))
print '\tConverting image class'
testImgClass = np_utils.to_categorical(testImgClass,10) #since we have 10 classes

print 'Classifying image'
classifEval = classif_model.evaluate(testImg, testImgClass)
accuracy = classifEval[1]
print '\tAccuracy:', accuracy

print '\nDone'