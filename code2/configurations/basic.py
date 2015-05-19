# import theano
# import theano.tensor as T
import lasagne as lag
from lasagne.layers import dnn
import fun


validate_every = 5
save_every = 5

num_chunks = 100
chunk_size = 16384
batch_size = 32
im_height = 96
im_width = 96

learning_rate = 0.003

# Conv2DLayer = nn.layers.Conv2DLayer
# MaxPool2DLayer = nn.layers.MaxPool2DLayer

Conv2DLayer = dnn.Conv2DDNNLayer
MaxPool2DLayer = dnn.MaxPool2DDNNLayer


def model():
    l_in = lag.layers.InputLayer((batch_size, 3, im_height, im_width))

    l1_conv = Conv2DLayer(l_in, num_filters=8, filter_size=(7, 7),
                          strides=(2, 2), border_mode='same',
                          W=lag.init.Orthogonal('relu'))

    l1_pool = MaxPool2DLayer(l1_conv, ds=(2, 2))

    l2_conv = Conv2DLayer(l1_pool, num_filters=16, filter_size=(3, 3),
                          border_mode='same', W=lag.init.Orthogonal('relu'))

    l2_pool = MaxPool2DLayer(l2_conv, ds=(2, 2))

    l3_conv = Conv2DLayer(l2_pool, num_filters=32, filter_size=(3, 3),
                          border_mode='same', W=lag.init.Orthogonal('relu'))

    l3_pool = MaxPool2DLayer(l3_conv, ds=(2, 2))

    l4 = lag.layers.DenseLayer(lag.layers.dropout(l3_pool, p=0.5),
                               num_units=256, W=lag.init.Orthogonal('relu'))

    l_out = lag.layers.DenseLayer(lag.layers.dropout(l4, p=0.5), num_units=1,
                                  nonlinearity=fun.scaled_sigmoid,
                                  W=lag.init.Orthogonal())
    # nonlinearity=nn_eyes.cutoff

    return l_in, l_out


# import theano
# import theano.tensor as T
# import lasagne as nn
# from lasagne.layers import dnn
# import nn_eyes
# import data
# import buffering


# validate_every = 10
# save_every = 10

# num_chunks = 4000
# chunk_size = 16384
# batch_size = 32
# im_height = 96
# im_width = 96

# learning_rate = 0.001

# augmentation_params = {
#     'zoom_range': (1 / 1.1, 1.1),
#     'rotation_range': (0, 360),
# }

# # Conv2DLayer = nn.layers.Conv2DLayer
# # MaxPool2DLayer = nn.layers.MaxPool2DLayer

# Conv2DLayer = dnn.Conv2DDNNLayer
# MaxPool2DLayer = dnn.MaxPool2DDNNLayer


# def build_model():
#     l_in = nn.layers.InputLayer((batch_size, 3, im_height, im_width))

#     l1_conv = Conv2DLayer(l_in, num_filters=16, filter_size=(7, 7), strides=(2, 2), border_mode='same', W=nn.init.Orthogonal('relu'))
#     l1_pool = MaxPool2DLayer(l1_conv, ds=(2, 2))

#     l2_conv = Conv2DLayer(l1_pool, num_filters=32, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
#     l2_pool = MaxPool2DLayer(l2_conv, ds=(2, 2))

#     l3_conv = Conv2DLayer(l2_pool, num_filters=64, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
#     l3_pool = MaxPool2DLayer(l3_conv, ds=(2, 2))

#     l4_conv = Conv2DLayer(l3_pool, num_filters=128, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
#     l4_pool = MaxPool2DLayer(l4_conv, ds=(2, 2))

#     l5 = nn.layers.DenseLayer(nn.layers.dropout(l4_pool, p=0.5), num_units=128, W=nn.init.Orthogonal('relu'))

#     l6 = nn.layers.DenseLayer(nn.layers.dropout(l5, p=0.5), num_units=128, W=nn.init.Orthogonal('relu'))

#     l_out = nn.layers.DenseLayer(nn.layers.dropout(l6, p=0.5), num_units=1, nonlinearity=None, W=nn.init.Orthogonal(), b=nn.init.Constant(2))

#     return l_in, l_out


# def minus_kappa(y, t):
#     return -nn_eyes.continuous_weighted_kappa(y[:, 0], t[:, 0])  # turn them back into 1D vectors


# def build_objective(l_in, l_out):
#     return nn.objectives.Objective(l_out, loss_function=minus_kappa)


# def create_train_gen():
#     image_gen = data.gen_images(data.paths_my_train, data.labels_my_train, shuffle=True, repeat=True)

#     def augmented_image_gen():
#         for image, label in image_gen:
#             yield data.augment_image(image, augmentation_params), label

#     chunks_gen = data.gen_chunks(augmented_image_gen(), chunk_size=chunk_size, labels=True)
#     return buffering.buffered_gen_threaded(chunks_gen)