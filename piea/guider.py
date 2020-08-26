

from __future__ import print_function
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Input
import keras.applications.mobilenet as mobilenet
#from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenet import preprocess_input

import keras.applications.inception_resnet_v2 as resnet
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_resnet_v2 import preprocess_input

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from keras_preprocessing import image
from PIL import Image
import scipy
import scipy.misc
import os
import numpy as np


class Guider():
    def __init__(self, inputs, guider, weight_file):
        self.inputs = inputs
        # self.cfg = cfg
        self.outputs = []
        self.si = np.arange(1, 11, 1)
        self.guider = guider
        self.weight_file = weight_file

        self.model()

    def model(self):
        weight_file = self.weight_file
        guider = self.guider

        #base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg')
        base_model = guider(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        #for layer in base_model.layers:
        #    layer.trainable = False
        #input = Input(self.input)
        #x = base_model(self.input)
        x = Dropout(1.)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)

        model.load_weights(weight_file)

        #if os.path.exists('weights/mobilenet_weights.h5'):
        #    model.load_weights('weights/mobilenet_weights.h5')

        self.outputs = []
        self.output_sd = []
        for input in self.inputs:
            inputlayer = Input(tensor=input)
            sd = model(inputlayer)
            output = K.sum( sd * self.si, axis=1)
            self.output_sd.append(sd)
            self.outputs.append(output)

        return self.outputs
