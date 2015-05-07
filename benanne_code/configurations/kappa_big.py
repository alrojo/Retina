import theano
import theano.tensor as T
import lasagne as nn
from lasagne.layers import dnn
import nn_eyes
import data
import buffering


validate_every = 10
save_every = 10

num_chunks = 4000
chunk_size = 16384
batch_size = 32
im_height = 96
im_width = 96

learning_rate = 0.001

# Conv2DLayer = nn.layers.Conv2DLayer
# MaxPool2DLayer = nn.layers.MaxPool2DLayer

Conv2DLayer = dnn.Conv2DDNNLayer
MaxPool2DLayer = dnn.MaxPool2DDNNLayer


def build_model():
    l_in = nn.layers.InputLayer((batch_size, 3, im_height, im_width))

    l1_conv = Conv2DLayer(l_in, num_filters=16, filter_size=(7, 7), strides=(2, 2), border_mode='same', W=nn.init.Orthogonal('relu'))
    l1_pool = MaxPool2DLayer(l1_conv, ds=(2, 2))

    l2_conv = Conv2DLayer(l1_pool, num_filters=32, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l2_pool = MaxPool2DLayer(l2_conv, ds=(2, 2))

    l3_conv = Conv2DLayer(l2_pool, num_filters=64, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l3_pool = MaxPool2DLayer(l3_conv, ds=(2, 2))

    l4 = nn.layers.DenseLayer(nn.layers.dropout(l3_pool, p=0.5), num_units=256, W=nn.init.Orthogonal('relu'))

    l_out = nn.layers.DenseLayer(nn.layers.dropout(l4, p=0.5), num_units=1, nonlinearity=None, W=nn.init.Orthogonal(), b=nn.init.Constant(2))

    return l_in, l_out


def minus_kappa(y, t):
    return -nn_eyes.continuous_weighted_kappa(y[:, 0], t[:, 0]) # turn them back into 1D vectors


def build_objective(l_in, l_out):
    return nn.objectives.Objective(l_out, loss_function=minus_kappa)
