from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import os
import pickle
from misc.utils import get_image, mkdir_p
import scipy.misc
import pandas as pd
import glob

if 'win' in sys.platform:
    delimeter = '\\'
elif 'linux' in sys.platform:
    delimeter = '/'
else:
    error('something wrong!!')

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
WEB_DIR = 'Data/web200'

# TODO: split train /test set!!
def save_filenames(data_dir, class_names):
    for mode in ['train','test']:
        tmp = glob.glob(os.path.join(data_dir,'images/*/*.jpg'))
        filenames = [s[s.rfind('images'+delimeter)+len('images'+delimeter):] for s in tmp]
        filenames = [s[:s.rfind('.jpg')] for s in filenames]
        class_info = [class_names.index(f[:f.rfind(delimeter)]) for f in filenames]
        outfile = data_dir + '/' + mode + '/filenames.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(filenames, f_out)
        outfile = data_dir + '/' + mode + '/class_info.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(class_info, f_out)

def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def save_data_list(inpath, outpath, filenames):
    hr_images = []
    lr_images = []
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    cnt = 0
    for key in filenames:
        f_name = '%s/images/%s.jpg' % (inpath, key)
        img = get_image(f_name, LOAD_SIZE, is_crop=False, bbox=None)
        img = img.astype('uint8')
        hr_images.append(img)
        lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic')
        lr_images.append(lr_img)
        cnt += 1
        if cnt % 100 == 0:
            print('Load %d......' % cnt)
    #
    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    #
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)


def convert_web_dataset_pickle(inpath):
    mkdir_p(os.path.join(inpath, 'train'))
    mkdir_p(os.path.join(inpath, 'test'))
    # class names
    tmp = glob.glob(os.path.join(inpath,'images','*'))
    class_names = [s[s.rfind(delimeter)+1:] for s in tmp]
    class_names.sort()
    print(class_names)
    # Save filenames
    save_filenames(inpath, class_names)

    # ## For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames)


if __name__ == '__main__':
    convert_web_dataset_pickle(WEB_DIR)
