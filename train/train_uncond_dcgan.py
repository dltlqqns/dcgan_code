EXP_NAME = ''
MODEL_NAME = 'uncond_dcgan'
DATASET = 'CUB_200_2011'  #'cifar10'
IMG_SIZE = 64
CLASSNAME = 'crop' #'ship'
LOAD_MODEL = '' #'64_cifar10_uncond_dcgan_horse_400'
BASE_COMPILEDIR = 'tmp/%s_%s_%s_%d'%(DATASET, CLASSNAME, MODEL_NAME, IMG_SIZE)
GPU_ID = 0
MODEL_DIR = 'models'
SAMPLES_DIR = 'samples'
LOSS_TYPE = 'GAN' #'GAN'

k_discrim = 1     # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 400*k_discrim        # # of iter at starting learning rate
niter_decay = 400*k_discrim   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
clip = 0.01

import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

os.environ['THEANO_FLAGS'] = 'base_compiledir=%s, device=gpu%d'%(BASE_COMPILEDIR, GPU_ID)
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
print("Base compile directory: %s"%theano.config.base_compiledir)

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from lib.metrics import nnc_score, nnd_score

from load import *

trX, vaX, teX, _, _, _ = load_uncond(DATASET, CLASSNAME, IMG_SIZE)
ntrain, nval, ntest = len(trX), len(vaX), len(teX)

vaX = floatX(vaX)/127.5 - 1.

def transform(X):
    #X = [center_crop(x, npx) for x in X]
    #return floatX(X).transpose(0, 3, 1, 2)/127.5 - 1.
    return (floatX(X)/127.5).reshape(-1, nc, npx, npx) - 1.

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0,2,3,1)+1)*127.5
    return X


desc = '%s_%s_%s_%s'%(EXP_NAME, DATASET, MODEL_NAME, CLASSNAME)
model_dir = '%s/%s'%(MODEL_DIR, desc)
samples_dir = '%s/%s'%(SAMPLES_DIR, desc)
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

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

dw  = difn((ndf, nc, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4')
db4 = bias_ifn((ndf*8), 'db4')
dwy = difn((ndf*8*4*4, 1), 'dwy')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    h4 = T.flatten(h4, 2)
    if LOSS_TYPE=='WGAN':
        y = T.dot(h4, wy)
    elif LOSS_TYPE=='GAN':
        y = sigmoid(T.dot(h4, wy))
    return y

X = T.tensor4()
Z = T.matrix()

gX = gen(Z, *gen_params)

p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

if LOSS_TYPE=='WGAN':
    d_cost_real = p_real.mean()
    d_cost_gen = -p_gen.mean()
    g_cost_d = p_gen.mean()
elif LOSS_TYPE=='GAN':
    d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
if LOSS_TYPE=='WGAN':
    pass
    #clip_updates = 
    #d_updater = updates.RMSprop(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    #g_updater = updates.RMSprop(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    #updates = d_updates + g_updates + clip_updates
elif LOSS_TYPE=='GAN':
    d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z], cost, updates=g_updates)
_train_d = theano.function([X, Z], cost, updates=d_updates)
_gen = theano.function([Z], gX)
if LOSS_TYPE=='WGAN':
    pass
    #_clip_d = theano.function([], updates=clip_updates)
print '%.2f seconds to compile theano functions'%(time()-t)

vis_idxs = py_rng.sample(np.arange(len(vaX)), 81)
vaX_vis = inverse_transform(vaX[vis_idxs])
color_grid_vis(vaX_vis, (9, 9), os.path.join(samples_dir, '%s_etl_test.png'%desc))

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(100, nz)))

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
    return np.concatenate(samples, axis=0)

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'g_cost',
    'd_cost',
]

vaX = vaX.reshape(len(vaX), -1)

# Load pretrained weight
if LOAD_MODEL!='':
    gen_load_path = '../pretrained/%s_gen_params.jl'%LOAD_MODEL
    discrim_load_path = '../pretrained/%s_discrim_params.jl'%LOAD_MODEL
    [p.set_value(v) for v,p in zip(joblib.load(gen_load_path), gen_params)]
    [p.set_value(v) for v,p in zip(joblib.load(discrim_load_path), discrim_params)]
    print("Model %s loaded!!"%LOAD_MODEL)


print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(niter+niter_decay+1):
    trX = shuffle(trX)
    for imb in tqdm(iter_data(trX, size=nbatch), total=ntrain/nbatch):
        imb = transform(imb)
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % (k_discrim+1) == 0:
            cost = _train_g(imb, zmb)
        else:
            cost = _train_d(imb, zmb)
            if LOSS_TYPE=='WGAN':
                pass
                #TODO: implement
        n_updates += 1
        n_examples += len(imb)
    if epoch%5 == 0:
        g_cost = float(cost[0])
        d_cost = float(cost[1])
        log = [n_epochs, n_updates, n_examples, time()-t, g_cost, d_cost]
        print '%.0f %.4f %.4f'%(epoch, g_cost, d_cost)
        f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
        f_log.flush()

    samples = np.asarray(_gen(sample_zmb))
    color_grid_vis(inverse_transform(samples), (10, 10), os.path.join(samples_dir, '%d.png'%(n_epochs)))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]:
        joblib.dump([p.get_value() for p in gen_params], os.path.join(model_dir, '%d_gen_params.jl'%n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], os.path.join(model_dir, '%d_discrim_params.jl'%n_epochs))

# Save last model
joblib.dump([p.get_value() for p in discrim_params], os.path.join(model_dir, '%d_discrim_params.jl'%epoch))
joblib.dump([p.get_value() for p in gen_params], os.path.join(model_dir, '%d_gen_params.jl'%epoch))
