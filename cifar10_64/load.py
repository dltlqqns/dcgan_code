import sys
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt
import pickle
import scipy.misc

from lib.data_utils import shuffle
from lib.config import data_dir

NUM_IMG_PER_CLASS = 500
NUM_CLASS = 10
IMG_SIZE = 64

def cifar10():
    ### train ###
    print("train")
    imgs = np.empty(shape=[0,3072])
    labels = np.empty(shape=[0])
    for i in range(1,6):
        filename = 'data_batch_%d'%i
        print(filename)
        with open(os.path.join(data_dir, filename),'rb') as f:
            tmp = pickle.load(f)
            sel = np.random.permutation(10000)[:NUM_IMG_PER_CLASS*2*6/5]
            imgs_sel = tmp['data'][sel]
            labels_sel = np.array(tmp['labels'])[sel]
            imgs = np.concatenate((imgs, imgs_sel), axis=0)
            labels = np.concatenate((labels, labels_sel), axis=0)
            labels = labels.astype(int)
            print('images size:')
            print(imgs.shape)
            print('labels:')
            print(len(labels))

    trX = imgs.reshape(-1,3,32,32).transpose((0,2,3,1))
    trX_re = [scipy.misc.imresize(img, [IMG_SIZE, IMG_SIZE]) for img in trX]
    trX = np.array(trX_re).transpose((0,3,1,2)).reshape(-1, IMG_SIZE*IMG_SIZE*3)
    trY = labels
    print("trX")
    print(trX.shape)
    print(type(trX))
    print("trY")
    print(trY.shape)
    print(type(trY))

    ### test ###
    print("test")
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        tmp = pickle.load(f)
        imgs = tmp['data']
        labels = tmp['labels']
    sel = np.random.permutation(10000)[:NUM_IMG_PER_CLASS*2]
    imgs = imgs[sel]
    labels = np.array(labels)
    labels = labels[sel]

    teX = imgs.reshape(-1,3,32,32).transpose((0,2,3,1))
    teX_re = [scipy.misc.imresize(img, [IMG_SIZE, IMG_SIZE]) for img in teX]
    teX = np.array(teX_re).transpose((0,3,1,2)).reshape(-1, IMG_SIZE*IMG_SIZE*3)
    teY = labels
    print("teX")
    print(teX.shape)
    print(type(teX))
    print("teY")
    print(teY.shape)
    print(type(teY))

    return trX, teX, trY, teY

def load_cifar10():
    trX, teX, trY, teY = cifar10()

    trX, trY = shuffle(trX, trY)
    vaX = trX[NUM_IMG_PER_CLASS*NUM_CLASS:]
    vaY = trY[NUM_IMG_PER_CLASS*NUM_CLASS:]
    trX = trX[:NUM_IMG_PER_CLASS*NUM_CLASS]
    trY = trY[:NUM_IMG_PER_CLASS*NUM_CLASS]

    return trX, vaX, teX, trY, vaY, teY
