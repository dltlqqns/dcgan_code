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
from lib.config import *

NUM_CLASS = 10
NUM_CHANNEL = 3

def cifar10(img_size, nsample_per_class):
    ### train ###
    print("train")
    imgs = np.empty(shape=[0,3072])
    labels = np.empty(shape=[0])
    for i in range(1,6):
        filename = 'data_batch_%d'%i
        print(filename)
        with open(os.path.join(data_dir_cifar10, filename),'rb') as f:
            tmp = pickle.load(f)
            sel = np.random.permutation(10000)[:nsample_per_class*2*6/5]
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
    trX_re = [scipy.misc.imresize(img, [img_size, img_size]) for img in trX]
    trX = np.array(trX_re).transpose((0,3,1,2)).reshape(-1, img_size*img_size*3)
    trY = labels
    print("trX")
    print(trX.shape)
    print(type(trX))
    print("trY")
    print(trY.shape)
    print(type(trY))

    ### test ###
    print("test")
    with open(os.path.join(data_dir_cifar10, 'test_batch'), 'rb') as f:
        tmp = pickle.load(f)
        imgs = tmp['data']
        labels = tmp['labels']
    sel = np.random.permutation(10000)[:nsample_per_class*2]
    imgs = imgs[sel]
    labels = np.array(labels)
    labels = labels[sel]

    teX = imgs.reshape(-1,3,32,32).transpose((0,2,3,1))
    teX_re = [scipy.misc.imresize(img, [img_size, img_size]) for img in teX]
    teX = np.array(teX_re).transpose((0,3,1,2)).reshape(-1, img_size*img_size*3)
    teY = labels
    print("teX")
    print(teX.shape)
    print(type(teX))
    print("teY")
    print(teY.shape)
    print(type(teY))

    return trX, teX, trY, teY

def web(img_size, nsample_per_class):
    ### train ###
    print("train")
    f = open(os.path.join(data_dir_web%(nsample_per_class), 'train','76images.pickle'),'rb')
    trX = pickle.load(f)
    trX_re = [scipy.misc.imresize(img, [img_size, img_size]) for img in trX]
    trX = np.array(trX_re).transpose((0,3,1,2)).reshape(-1, img_size*img_size*NUM_CHANNEL)
    f = open(os.path.join(data_dir_web%(nsample_per_class), 'train','class_info.pickle'),'rb')
    trY = pickle.load(f)
    trY = np.array(trY)
    print("trX")
    print(trX.shape)
    print(type(trX))
    print("trY")
    print(trY.shape)
    print(type(trY))

    ### test ###
    print("test")
    f = open(os.path.join(data_dir_web%(nsample_per_class), 'test','76images.pickle'),'rb')
    teX = pickle.load(f)
    teX_re = [scipy.misc.imresize(img, [img_size, img_size]) for img in teX]
    teX = np.array(teX_re).transpose((0,3,1,2)).reshape(-1, img_size*img_size*NUM_CHANNEL)
    f = open(os.path.join(data_dir_web%(nsample_per_class), 'test','class_info.pickle'),'rb')
    teY = pickle.load(f)
    teY = np.array(teY)
    print("teX")
    print(teX.shape)
    print(type(teX))
    print("teY")
    print(teY.shape)
    print(type(teY))

    return trX, teX, trY, teY

def load_cifar10(img_size, nsample_per_class):
    trX, teX, trY, teY = cifar10(img_size, nsample_per_class)

    trX, trY = shuffle(trX, trY)
    vaX = trX[nsample_per_class*NUM_CLASS:]
    vaY = trY[nsample_per_class*NUM_CLASS:]
    trX = trX[:nsample_per_class*NUM_CLASS]
    trY = trY[:nsample_per_class*NUM_CLASS]

    return trX, vaX, teX, trY, vaY, teY

def load_web(img_size, nsample_per_class):
    trX, teX, trY, teY = web(img_size, nsample_per_class)
    trX, trY = shuffle(trX, trY)

    boundary = int(np.floor(trX.shape[0]*0.9))
    vaX = trX[boundary:]
    vaY = trY[boundary:]
    trX = trX[:boundary]
    trY = trY[:boundary]

    return trX, vaX, teX, trY, vaY, teY
