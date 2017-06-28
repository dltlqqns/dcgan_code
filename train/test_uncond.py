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

IMG_SIZE = 64
nc = 3            # # of channels in image
npx = IMG_SIZE          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 1024         # # of gen filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
LOAD_GEN_PATHS = ['./models/_google_car_uncond_dcgan_truck_tmp/200_gen_params.jl', \
                  './models/_google_car_uncond_dcgan_truck_tmp/400_gen_params.jl', \
                  './models/_google_car_uncond_dcgan_truck_tmp/600_gen_params.jl', \
                  './models/_google_car_uncond_dcgan_truck_tmp/800_gen_params.jl']
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
gw  = gifn((nz, ngf*4*4), 'gw')
gg = gain_ifn((ngf*4*4), 'gg')
gb = bias_ifn((ngf*4*4), 'gb')
gw2 = gifn((ngf, ngf/2, 5, 5), 'gw2')
gg2 = gain_ifn((ngf/2), 'gg2')
gb2 = bias_ifn((ngf/2), 'gb2')
gw3 = gifn((ngf/2, ngf/4, 5, 5), 'gw3')
gg3 = gain_ifn((ngf/4), 'gg3')
gb3 = bias_ifn((ngf/4), 'gb3')
gw4 = gifn((ngf/4, ngf/8, 5, 5), 'gw4')
gg4 = gain_ifn((ngf/8), 'gg4')
gb4 = bias_ifn((ngf/8), 'gb4')
ng_final = ngf/8
if IMG_SIZE>=128:
    gw5 = gifn((ngf/8, ngf/16, 5, 5), 'gw5')      #128
    gg5 = gain_ifn((ngf/16), 'gg5')                #128
    gb5 = bias_ifn((ngf/16), 'gb5')                #128
    ng_final = ngf/16
    if IMG_SIZE==256:
        gw6 = gifn((ngf/16, ngf/32, 5, 5), 'gw6')    #256
        gg6 = gain_ifn((ngf/32), 'gg6')                #256
        gb6 = bias_ifn((ngf/32), 'gb6')                #256
        ng_final = ngf/32
gwx = gifn((ng_final, nc, 5, 5), 'gwx')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
if IMG_SIZE>=128:
    gen_params.insert(-1, gw5)
    gen_params.insert(-1, gg5)
    gen_params.insert(-1, gb5)
    if IMG_SIZE==256:    
        gen_params.insert(-1, gw6)
        gen_params.insert(-1, gg6)
        gen_params.insert(-1, gb6)

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
#def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, wx):
#def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    if IMG_SIZE==64:
        x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    elif IMG_SIZE==128:
        h5 = relu(batchnorm(deconv(h4, w5, subsample=(2, 2), border_mode=(2, 2)), g=g5, b=b5)) #128
        x = tanh(deconv(h5, wx, subsample=(2, 2), border_mode=(2, 2)))
    elif IMG_SIZE==256:
        h5 = relu(batchnorm(deconv(h4, w5, subsample=(2, 2), border_mode=(2, 2)), g=g5, b=b5)) #128
        h6 = relu(batchnorm(deconv(h5, w6, subsample=(2, 2), border_mode=(2, 2)), g=g6, b=b6)) #256
        x = tanh(deconv(h6, wx, subsample=(2, 2), border_mode=(2, 2)))
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
