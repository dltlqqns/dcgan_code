import sys
sys.path.append('..')
import json
import numpy as np
import theano
import theano.tensor as T
from lib import activations
from lib import inits
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from sklearn.externals import joblib
from inception_score import get_inception_score

nc = 3            # # of channels in image
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
LOAD_GEN_PATHS = ['./models/_CUB_200_2011_uncond_dcgan_/200_gen_params.jl', \
                  './models/_CUB_200_2011_uncond_dcgan_/400_gen_params.jl', \
                  './models/_CUB_200_2011_uncond_dcgan_/600_gen_params.jl', \
                  './models/_CUB_200_2011_uncond_dcgan_/800_gen_params.jl']
NUM_SAMPLE = 500


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()

bce = T.nnet.binary_crossentropy
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)
gw  = gifn((nz, ngf*8*4*4), 'gw')
gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb')
gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4')
gwx = gifn((ngf, nc, 5, 5), 'gwx')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x
    
Z = T.matrix()
gX = gen(Z, *gen_params)
_gen = theano.function([Z], gX)

def gen_samples(n, nbatch=128):
    samples = []
    n_gen = 0
    for i in range(n/nbatch):
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb)
        samples.append(xmb)
        n_gen += len(xmb)
    n_left = n-n_gen
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb)
    samples.append(xmb)    
    samples = np.concatenate(samples, axis = 0)
    samples = np.uint8((np.transpose(samples, (0,2,3,1))+1)*127.5)
    return samples

# Load generator weight
for load_gen_path in LOAD_GEN_PATHS:
    [p.set_value(v) for v,p in zip(joblib.load(load_gen_path), gen_params)]
    print("Model is loaded from %s!!"%load_gen_path)

    # test
    samples = gen_samples(NUM_SAMPLE)
    samples = [sample for sample in samples]
    score, score_std = get_inception_score(samples)
    print('---score: %f, score_std: %f'%(score, score_std))