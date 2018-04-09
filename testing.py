from keras.models import Model, load_model, model_from_json
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
import cv2
import numpy as np
import os
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove tensorflow verbose 

print '\n__________________________________\n'
print 'Loading Classification Model\n\n\n'

# load json and create model
json_file = open('cifar10classification_top5.json', 'r')
classif_model_json = json_file.read()
json_file.close()
classif_model = model_from_json(classif_model_json)
# load weights into new model
classif_model.load_weights("cifar10classification_top5.h5")
print("Loaded model from disk")

epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classif_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#
# Loading training images
#

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

y_train= np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print 'Loaded', len(x_train), 'clean training images, shape:', np.array(x_train).shape
print 'Loaded', len(x_test), 'clean testing images, shape:', np.array(x_test).shape

print '\nClassifying ground truth images...'
x_test_classifEval = classif_model.evaluate(x_test, y_test, verbose=0)
x_test_classifAccuracy = x_test_classifEval[1]
print x_test_classifAccuracy

print '\nClassifying noisy images...'
x_train_noisy_classifEval = classif_model.evaluate(x_train_noisy, y_train, verbose=0)
x_train_noisy_classifAccuracy = x_train_noisy_classifEval[1]
print x_train_noisy_classifAccuracy