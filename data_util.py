import os
import glob
import numpy as np
from keras.preprocessing.image import load_img

def data_list(base):
    subfolders = glob.glob(base)
    train_data = []
    for subfolder in subfolders[: -1]:
    images = np.array(glob.glob(os.path.join(subfolder,'*.jpg')))
    train_data.append(images.tolist())
    return train_data

def read_image(datalist):
    imgdata = []
    for imgname in datalist:
        img = load_img(imgname,target_size = (64,64))
        imgarray = np.asarray(img)
        imgarray -= np.mean(imgarray)
        imgarray /= 128.0
        imgarray.reshape((1,)+imgarray.shape)
        imgdata.append(imgarray)

    return np.vstack(imgdata)
