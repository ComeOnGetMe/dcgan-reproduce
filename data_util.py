import os
import glob
import numpy as np
import struct
from array import array as pyarray
from keras.preprocessing.image import load_img,img_to_array


def data_list(base):
    subfolders = glob.glob(base)
    train_data = []
    subfolder = subfolders[0]
    print subfolder
    images = np.array(glob.glob(os.path.join(subfolder,'*.webp')))
    return images

def read_image(datalist):
    imgdata = []
    for imgname in datalist:
        img = img_to_array(load_img(imgname,target_size= [64,64]))
        # img = img[img.shape[0]/2 - 32:img.shape[0]/2 + 32, img.shape[1]/2 - 32:img.shape[1]/2 + 32]
        imgarray = img / 127.5 - 1
        imgdata.append(np.reshape(imgarray,(1,)+imgarray.shape))

    return np.vstack(imgdata)
