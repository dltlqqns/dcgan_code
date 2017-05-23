import sys
sys.path.append('..')

import numpy as np
import os
import pickle
from lib.config import data_dir_cifar10
import errno
import scipy.misc

SAVE_IMG_SIZE = 32
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
delimeter = '_'
SAVE_DIR = 'cifar10_one'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_images(imgs, names):
    for idx, img in enumerate(imgs):
        filepath = os.path.join(SAVE_DIR, names[idx]+'.jpg')
        mkdir_p(os.path.dirname(filepath))
        if img.shape[0] != SAVE_IMG_SIZE:
            img = scipy.misc.imresize(img, [SAVE_IMG_SIZE, SAVE_IMG_SIZE])
        scipy.misc.imsave(filepath, img)


### train ###
for i in range(1,6):
    filename = 'data_batch_%d'%i
    with open(os.path.join(data_dir_cifar10, filename), 'rb') as f:
        tmp = pickle.load(f)
        imgs = tmp['data'].reshape(-1,3,32,32).transpose((0,2,3,1))
        names = [CLASS_NAMES[u] + delimeter + v for u,v in zip(tmp['labels'], tmp['filenames'])]
        print(imgs.shape)
        save_images(imgs, names)

