from model import *
import tensorflow as tf
from data_util import *

data = read_image(data_list('../lsun/images/classroom'))
print data.shape

with tf.Session() as sess:
    dcgan = DCGan(sess)
    dcgan.fit(10,data)
