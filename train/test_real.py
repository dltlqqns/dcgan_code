import glob
import sys
sys.path.append('..')
from lib.data_utils import shuffle
import scipy.misc
from inception_score import get_inception_score

img_list = glob.glob('/home/yumin/dataset/web_car/*/*.jpg')
img_list = shuffle(img_list)[:500]
samples = [scipy.misc.imread(v) for v in img_list]
samples = [scipy.misc.imresize(v, (224,224)) for v in samples]
score, score_std = get_inception_score(samples)
print('score: %f, score_std: %f'%(score, score_std))
