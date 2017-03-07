from model import generator_model, discriminator_model,DCGan
from keras.layers import Lambda, Merge, Input
from keras.models import Sequential
import numpy as np


dcgan = DCGan(100,[])

dcgan.train_model.summary()

dcgan.test_model.summary()
