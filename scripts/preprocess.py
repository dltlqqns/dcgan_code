import sys
sys.path.append('..')
import os
import glob
import scipy.misc
import pickle
from utils import mkdir_p

DATASET = 'web5000'
ROOT_DIR = '/home/yumin/dataset/%s/'%DATASET
CLASSNAME = 'horse'
IMG_SIZE = 64

def convert_dataset_pickle(root_dir, dataset, classname, img_size):
    out_dir = os.path.join('..', 'Data', dataset)
    mkdir_p(out_dir)
    print("save dataset to %s"%out_dir)
    filenames = glob.glob(os.path.join(root_dir, classname, '*.jpg'))
    print("#img: " + str(len(filenames)))

    imgs = []
    for filename in filenames:
        img = scipy.misc.imread()
        img = img.astype('uint8')
        img = scipy.misc.imresize(img, [img_size, img_size], 'bicubic')
        imgs.append(img)
        
    filenames = [os.path.basename(v) for v in filenames]
    outfile = os.path.join(out_dir, 'filenames_%s.pickle'%(classname))
    with open(outfile, 'wb') as f_out:
        pickle.dump(filenames, f_out)
    outfile = os.path.join(out_dir, '%dimages_%s.pickle'%(img_size, classname))
    with open(outfile, 'wb') as f_out:
        pickle.dump(imgs, f_out)

if __name__ == '__main__':
    convert_dataset_pickle(ROOT_DIR, DATASET, CLASSNAME, IMG_SIZE)
