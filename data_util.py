import os
import glob
import numpy as np
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
    for imgname in datalist[:1024]:
        img = load_img(imgname,target_size = (64,64))
        imgarray = img_to_array(img)/ 255
        imgarray = imgarray * 2 - 1
        imgdata.append(np.reshape(imgarray,(1,)+imgarray.shape))

    return np.vstack(imgdata)
