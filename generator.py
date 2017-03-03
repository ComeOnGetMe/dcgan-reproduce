import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Deconvolution2D
from keras import initializations
import numpy as np


def gaussian(shape, name=None,dim_ordering=None):
    return initializations.normal(shape, scale=0.02, name=name)

def generator(input_dim):
    '''
    Taking uniformly distributed random vectors as input, and generating
    64x64 images to confuse the discriminator. All the structures are the
    same as the one mentioned in the DCGan paper.
    '''

    x_in = Input(shape = (input_dim,))

    # FC Layer
    x_full = Dense(1024 * 4 * 4, activation = 'relu', init = gaussian)(x_in)

    # Reshape the vector to a 4-d tensor (Batch dimension is omitted)
    conv_in = Reshape((4, 4, 1024))(x_full)
    conv_in = BatchNormalization()(conv_in)

    # Transpose convolution layer 1, (batch, 4, 4, 1024) -> (batch, 8, 8, 512)
    tconv1 = Deconvolution2D(512, 5, 5,
                        subsample = (1,1),
                        activation = 'relu',
                        init = gaussian,
                        output_shape=(None, 8, 8, 512))(conv_in)
    tconv1 = BatchNormalization()(tconv1)

    # Transpose convolution layer 2, (batch, 8, 8, 512) -> (batch, 16, 16, 256)
    tconv2 = Deconvolution2D(256,5,5,
                        subsample = (2,2),
                        activation = 'relu',
                        init = gaussian,
                        border_mode = 'same',
                        output_shape = (None, 16, 16, 256))(tconv1)
    tconv2 = BatchNormalization()(tconv2)

    # Transpose convolution layer 3, (batch, 16, 16, 256) -> (batch, 32, 32, 128)
    tconv3 = Deconvolution2D(128,5,5,
                        subsample = (2,2),
                        activation = 'relu',
                        init = gaussian,
                        border_mode = 'same',
                        output_shape = (None, 32, 32, 128))(tconv2)
    tconv3 = BatchNormalization()(tconv3)

    # Transpose convolution layer 4, (batch, 32, 32, 128) -> (batch, 64, 64, 3)
    tconv4 = Deconvolution2D(3,5,5,
                        subsample = (2,2),
                        activation = 'tanh',
                        init = gaussian,
                        border_mode = 'same',
                        output_shape = (None, 64, 64, 3))(tconv3)

    return Model(input = x_in, output = tconv4)
