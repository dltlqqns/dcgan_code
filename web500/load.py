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

IMG_SIZE = 64
NUM_CHANNEL = 3
data_dir = '/home/yumin/codes/StackGAN/Data/web200'

def web500():
    ### train ###
    print("train")
    f = open(os.path.join(data_dir, 'train','76images.pickle'),'rb')
    trX = pickle.load(f)
    trX_re = [scipy.misc.imresize(img, [IMG_SIZE, IMG_SIZE]) for img in trX]
    trX = np.array(trX_re).transpose((0,3,1,2)).reshape(-1, IMG_SIZE*IMG_SIZE*NUM_CHANNEL)
    f = open(os.path.join(data_dir, 'train','class_info.pickle'),'rb')
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
    f = open(os.path.join(data_dir, 'test','76images.pickle'),'rb')
    teX = pickle.load(f)
    teX_re = [scipy.misc.imresize(img, [IMG_SIZE, IMG_SIZE]) for img in teX]
    teX = np.array(teX_re).transpose((0,3,1,2)).reshape(-1, IMG_SIZE*IMG_SIZE*NUM_CHANNEL)
    f = open(os.path.join(data_dir, 'test','class_info.pickle'),'rb')
    teY = pickle.load(f)
    teY = np.array(teY)
    print("teX")
    print(teX.shape)
    print(type(teX))
    print("teY")
    print(teY.shape)
    print(type(teY))

    return trX, teX, trY, teY

def load_web500():
    trX, teX, trY, teY = web500()
    trX, trY = shuffle(trX, trY)

    boundary = int(np.floor(trX.shape[0]*0.9))
    vaX = trX[boundary:]
    vaY = trY[boundary:]
    trX = trX[:boundary]
    trY = trY[:boundary]

    return trX, vaX, teX, trY, vaY, teY
