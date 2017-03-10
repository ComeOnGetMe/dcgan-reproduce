from model import *
import tensorflow as tf
from data_util import *



with tf.Session() as sess:
    data = read_image(data_list('../256_ObjectCategories'),25600)
    dcgan = DCGan(sess)
    dcgan.fit(10,data)
