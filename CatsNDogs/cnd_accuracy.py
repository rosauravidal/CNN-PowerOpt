import cv2
import os
import sys
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove tensorflow verbose 

modelLocation = sys.argv[1] # Baseline.h5
print 'Loading model...', modelLocation
model = load_model(modelLocation)

imgDir = 'data/testing/'

gen = image.ImageDataGenerator()
batch_size = 100


test_batches = gen.flow_from_directory(imgDir,
	target_size=(224,224),
	class_mode='categorical',
	shuffle=True,
	batch_size=batch_size)

n = test_batches.next()
imgs = np.array(n[0])
labels = np.array(n[1])


#labels = np_utils.to_categorical(labels,2) #since we have 2 classes [cat, dog]

print 'Accuracy', model.evaluate(imgs, labels)[1]