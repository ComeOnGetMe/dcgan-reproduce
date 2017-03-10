import os
import glob
import numpy as np
from keras.preprocessing.image import load_img

def data_list(base):
    subfolders = glob.glob(base)
    train_data = []
    subfolder in subfolders[: -1]:
    images = np.array(glob.glob(os.path.join(subfolder,'*.jpg')))
    return images

def read_image(datalist,num):
    ind = np.random.permutation(datalist.shape[0])[:num]
    imgdata = []
    i = 0
    while i < num:
        img = load_img(datalist[ind[i]],target_size = (64,64))
        imgarray = np.copy(np.asarray(img))
        imgarray -= np.mean(imgarray)
        imgarray /= 128.0
        imgarray.reshape((1,)+imgarray.shape)
        imgdata.append(imgarray)

    return np.vstack(imgdata)
