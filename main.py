from model import *
import tensorflow as tf
from data_util import *

data = read_image(data_list('../lsun/images/church'))
# data = mnist_read(3)
print data.shape

with tf.Session() as sess:
    dcgan = DCGan(sess)
    dcgan.fit(50,data)
