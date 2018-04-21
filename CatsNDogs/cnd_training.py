import os
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten	
from keras.models import Model, load_model
from keras import optimizers

import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove tensorflow verbose 

trainDir = 'data/train/'
validDir = 'data/validation/'

gen = image.ImageDataGenerator()
batch_size = 64


train_batches = gen.flow_from_directory(trainDir,
	target_size=(224,224),
	class_mode='categorical',
	shuffle=True,
	batch_size=batch_size)

val_batches = gen.flow_from_directory(validDir,
	target_size=(224,224),
	class_mode='categorical',
	shuffle=True,
	batch_size=batch_size)

#
# BASELINE
#

# #load full vgg model and add new FC to reduce preds dimensions from 1000 to 2
# vgg = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None)

# # make the rest of the model not trainable
# for layer in vgg.layers: layer.trainable=False

# # define a new output layer to connect with the last fc layer in vgg
# x = vgg.layers[-2].output # output of the second last layer
# output_layer = Dense(2, activation='softmax', name='predictions')(x)

# # combine original model and new output layer
# vgg2 = Model(inputs=vgg.input, outputs=output_layer)


#
# Net1  (FC : 256, Dropout, FC -output- 2)
#

# vgg = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None)

# for layer in vgg.layers: layer.trainable=False
# x = vgg.layers[-1].output
# fc = Flatten()(x)
# fc = Dense(256, activation='relu')(fc)
# fc = Dropout(0.5)(fc)
# fc = Dense(2, activation='softmax', name='predictions')(fc)

# vgg2 = Model(inputs=vgg.input, outputs=fc)


#
# Net2  (FC : 4096, Dropout, FC : 4096, FC -output- 2)
#


# vgg = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None)

# for layer in vgg.layers: layer.trainable=False

# x = vgg.layers[-3].output
# fc = Dropout(0.5)(x)
# fc = Dense(4096, activation='relu')(fc)
# fc = Dense(2, activation='softmax', name='predictions')(fc)

# fc[1].trainable=False

# vgg2 = Model(inputs=vgg.input, outputs=fc)

#
# Net3  (FC : 256, Dropout, FC -output- 2)
#

vgg = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None)

for layer in vgg.layers: layer.trainable=False
x = vgg.layers[-1].output
fc = Flatten()(x)
fc = Dense(128, activation='relu')(fc)
fc = Dropout(0.5)(fc)
fc = Dense(128, activation='relu')(fc)
fc = Dense(2, activation='softmax', name='predictions')(fc)

vgg2 = Model(inputs=vgg.input, outputs=fc)


print vgg2.summary()

with open('Net3_summary.txt','w') as f:
    vgg2.summary(print_fn=lambda x: f.write(x + '\n'))

# compile
vgg2.compile(optimizer=optimizers.SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#run it
vgg2.fit_generator(train_batches,
                   steps_per_epoch = train_batches.samples // batch_size,
                   validation_data = val_batches, 
                   validation_steps = val_batches.samples // batch_size,
                   epochs = 20)



vgg2.save('Net3.h5')