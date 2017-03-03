from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import initializations
import numpy as np

def gaussian(shape, name=None,dim_ordering=None):
    return initializations.normal(shape, scale=0.02, name=name)

def discriminator():
    '''
    The discriminator takes an image, sampled from the dataset or generated by
    the generator, as input, and outputs the probability if the input is a real
    sample in the dataset.
    '''

    x_in = Input(shape = (64,64,3))

    # Convolution layer 1, no BatchNorm according to the paper
    conv1 = Convolution2D(64,5,5,
                          border_mode = 'same',
                          subsample = (2,2),
                          init = gaussian)(x_in)
    conv1 = LeakyReLU(alpha = 0.2)(conv1)

    # Convolution layer 2
    conv2 = Convolution2D(128,5,5,
                          border_mode = 'same',
                          subsample = (2,2),
                          init = gaussian)(conv1)
    conv2 = LeakyReLU(alpha = 0.2)(conv2)
    conv2 = BatchNormalization()(conv2)

    # Convolution layer 3
    conv3 = Convolution2D(256,5,5,
                          border_mode = 'same',
                          subsample = (2,2),
                          init = gaussian)(conv2)
    conv3 = LeakyReLU(alpha = 0.2)(conv3)
    conv3 = BatchNormalization()(conv3)

    # Convolution layer 4
    conv4 = Convolution2D(512,5,5,
                          border_mode = 'same',
                          subsample = (2,2),
                          init = gaussian)(conv3)
    conv4 = LeakyReLU(alpha = 0.2)(conv4)
    conv4 = BatchNormalization()(conv4)

    # Flatten the convolution output, feed it into fc layer
    fc_in = Flatten()(conv4)
    out = Dense(1,init = gaussian, activation = 'sigmoid')(fc_in)

    return Model(input = x_in, output = out)

model = discriminator()
print model.summary()
