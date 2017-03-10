import os
import glob
import numpy as np
import struct
from array import array as pyarray
from keras.preprocessing.image import load_img,img_to_array

def mnist_read(digit):
    fimg = open("dataset/train-images.idx3-ubyte", 'r+b')
    flabel = open("dataset/train-labels.idx1-ubyte",'r+b')
    # timg = open("dataset/t10k-images.idx3-ubyte", 'r+b')
    # tlabel = open("dataset/t10k-labels.idx1-ubyte",'r+b')

    magic, trsize = struct.unpack(">II", flabel.read(8))
    magic, trsize, rows, cols = struct.unpack('>IIII', fimg.read(16))
    trlabel = pyarray("B", flabel.read())
    trimg = pyarray("B", fimg.read())
    flabel.close()
    fimg.close()

    # magic, tesize = struct.unpack(">II", tlabel.read(8))
    # magic, tesize, rows, cols = struct.unpack('>IIII', timg.read(16))
    # telabel = pyarray("B", tlabel.read())
    # teimg = pyarray("B", timg.read())
    # tlabel.close()
    # timg.close()

    train_label = np.zeros((trsize), dtype=np.uint8)
    train_img = np.zeros((trsize, rows * cols), dtype=np.float)
    for i in range(trsize):
        train_img[i] = np.array(trimg[i * rows * cols: (i + 1) * rows * cols])
        train_label[i] = trlabel[i]

    # test_label = np.zeros((tesize), dtype=np.float)
    # test_img = np.zeros((tesize, rows * cols), dtype=np.float)
    # for i in range(tesize):
    #     test_img[i] = np.array(teimg[i * rows * cols: (i + 1) * rows * cols])
    #     test_label[i] = telabel[i]

    train_image = np.hstack((np.ones((trsize,1)),train_img))/255.0
    train_image = np.reshape(train_image, train_image.shape + (1,)) * 2 - 1

    return train_image[train_label == digit]

    # test_image = np.hstack((np.ones((tesize,1)),test_img))/255.0
    # test_image = test_image[:2000]
    # te_label = test_label[:2000]

def data_list(base):
    subfolders = glob.glob(base)
    train_data = []
    subfolder = subfolders[0]
    print subfolder
    images = np.array(glob.glob(os.path.join(subfolder,'*.webp')))
    return images

def read_image(datalist):
    imgdata = []
    for imgname in datalist[:4096]:
        img = load_img(imgname,target_size = (64,64))
        imgarray = img_to_array(img)/ 255
        imgarray = imgarray * 2 - 1
        imgdata.append(np.reshape(imgarray,(1,)+imgarray.shape))

    return np.vstack(imgdata)
