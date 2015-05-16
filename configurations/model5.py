
import theano
import theano.tensor as T
import lasagne as nn
from lasagne.layers import dnn
import nn_eyes
import data
import buffering
import tmp_dnn
import numpy as np
import customLayers


meanAndStd = np.load('data/meanAndStd_256.npy')

probOut = True
validate_every = 10
save_every = 10

num_chunks = 4000
chunk_size = 3200
batch_size = 32
im_height = 256
im_width = 256
image_size = (im_height, im_width)
color_pertube = False

# Set momentum hyper parameter to 0.9
momentum = 0.9

# Specify learning rate decay during training
learning_rate_schedule = {
    0: 0.003,
    700: 0.0003,
    800: 0.00003,
}

augmentation_params = {
    'zoom_range': (1 / 1.2, 1.2),
    'rotation_range': (0, 40),
    'shear_range': (-5, 5),
    'translation_range': (-5, 5),
    'do_flip': False
}



Conv2DLayer = nn.layers.dnn.Conv2DDNNLayer
MaxPool2DLayer = nn.layers.dnn.MaxPool2DDNNLayer

# Conv2DLayer = dnn.Conv2DDNNLayer
# MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer


def build_model():
    l_in = nn.layers.InputLayer((batch_size, 3, im_height, im_width))

    l1a_c = Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l1b_c = Conv2DLayer(l1a_c, num_filters=16, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l1_pool = MaxPool2DLayer(l1b_c, (2, 2), (2, 2))

    l2a_c = Conv2DLayer(l1_pool, num_filters=64, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l2b_c = Conv2DLayer(l2a_c, num_filters=32, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l2_pool = MaxPool2DLayer(l2b_c, (2, 2), (2, 2))

    l3a_c = Conv2DLayer(l2_pool, num_filters=128, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l3b_c = Conv2DLayer(l3a_c, num_filters=64, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l3_pool = MaxPool2DLayer(l3b_c, (2, 2), (2, 2))

    l4a_c = Conv2DLayer(l3_pool, num_filters=256, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l4b_c = Conv2DLayer(l4a_c, num_filters=128, filter_size=(3, 3), border_mode='same', W=nn.init.Orthogonal('relu'))
    l4_pool = MaxPool2DLayer(l4b_c, (2, 2), (2, 2))

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4_pool, p=0.5), num_units=256, W=nn.init.Orthogonal('relu'))

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5, p=0.5), num_units=256, W=nn.init.Orthogonal('relu'))

    l_out = nn.layers.DenseLayer(nn.layers.dropout(l6, p=0.5), num_units=5, nonlinearity=T.nnet.softmax, W=nn.init.Orthogonal('relu'))

    return l_in, l_out


def build_objective(l_in, l_out):

    lambda_reg = 0.0005
    params = nn.layers.get_all_non_bias_params(l_out)
    reg_term = sum(T.sum(p**2) for p in params)

    def loss(x, t):
        return customLayers.mse(x, t) + lambda_reg * reg_term

    return nn.objectives.Objective(l_out, loss_function=loss)


def create_train_gen():
    # Generator function that yields a new image and corresponding label when it's called
    image_gen = data.gen_images(data.paths_my_train, meanAndStd, data.labels_my_train, shuffle=True, repeat=True)

    # Generator function that takes the image generated by image_gen and transform according to the parameters
    # set in "augmentation_params" or according to the default parameters specified inside the functions.
    def augmented_image_gen():
        for image, label in image_gen:
            yield data.augment_image(image, image_size, color_pertube, augmentation_params), label

    # Generator that generates chunk of data where the images have transformed by the function augmented_image_gen
    chunks_gen = data.gen_chunks(augmented_image_gen(), image_size=image_size, chunk_size=chunk_size, labels=True)
    return buffering.buffered_gen_threaded(chunks_gen)


def build_model():
    # variable scale part
    l0_variable = nn.layers.InputLayer((batch_size, 1, patch_sizes[0][0], patch_sizes[0][1]))
    l0c = dihedral.CyclicSliceLayer(l0_variable)

    l1a = Conv2DLayer(l0c, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1b = Conv2DLayer(l1a, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))

    l2a = Conv2DLayer(l1, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2b = Conv2DLayer(l2a, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))

    l3a = Conv2DLayer(l2, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3b = Conv2DLayer(l3a, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3c = Conv2DLayer(l3b, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))

    l4a = Conv2DLayer(l3, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l4b = Conv2DLayer(l4a, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l4c = Conv2DLayer(l4b, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l4 = MaxPool2DLayer(l4c, ds=(3, 3), strides=(2, 2))
    l4f = nn.layers.flatten(l4)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4f, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l5r = dihedral.CyclicRollLayer(l5)

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5r, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l_variable = dihedral.CyclicPoolLayer(l6, pool_function=nn_plankton.rms)


    # fixed scale part
    l0_fixed = nn.layers.InputLayer((batch_size, 1, patch_sizes[1][0], patch_sizes[1][1]))
    l0c = dihedral.CyclicSliceLayer(l0_fixed)

    l1a = Conv2DLayer(l0c, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1b = Conv2DLayer(l1a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))

    l2a = Conv2DLayer(l1, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2b = Conv2DLayer(l2a, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))

    l3a = Conv2DLayer(l2, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3b = Conv2DLayer(l3a, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3c = Conv2DLayer(l3b, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1))
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3f = nn.layers.flatten(l3)

    l4 = nn.layers.DenseLayer(nn.layers.dropout(l3f, p=0.5), num_units=128, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l4r = dihedral.CyclicRollLayer(l4)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4r, p=0.5), num_units=128, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1))
    l_fixed = dihedral.CyclicPoolLayer(l5, pool_function=nn_plankton.rms)


    # merge the parts
    l_merged = nn.layers.concat([l_variable, l_fixed])

    l7 = nn.layers.DenseLayer(nn.layers.dropout(l_merged, p=0.5), num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0_variable, l0_fixed], l7


