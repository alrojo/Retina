import numpy as np 
import skimage.transform
import skimage.io
import pandas as pd
import os
import glob

split = np.load("/home/josca/myalex/data/split.pkl")

csv = pd.read_csv("/home/josca/myalex/data/trainLabels.csv")
filenames_train = np.array(csv['image'])
labels_train = np.array(csv['level'])
labels_my_train = labels_train[split['train']]
labels_my_valid = labels_train[split['valid']]

num_images = len(labels_train)
# num_classes = 5

# ../../data/training/outimages512
# /home/ubuntu/data/training/outimages512

paths_train = np.array([os.path.join("/home/josca/myalex/train_256", "%s.png" % filename) for filename in filenames_train])

paths_my_train = paths_train[split['train']]
paths_my_valid = paths_train[split['valid']]

#paths_test = glob.glob("/home/josca/davidData/Retina/David/test_512*.png")
#paths_test.sort()
#paths_test = np.array(paths_test)


def gen_images(paths, labels=None, shuffle=False, repeat=False):
    paths_shuffled = np.array(paths)          # Load list of paths into numpy array

    if labels is not None:
        labels_shuffled = np.array(labels)    # Load list of labels into numpy array

    while True:
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(paths_shuffled)      # Array of paths is shuffled
            if labels is not None:
                # Note that the array of states and the array of paths are shuffled in the exact same way
                # since the random number generator has the same "inner state"
                np.random.set_state(state)
                np.random.shuffle(labels_shuffled)

        # Load image and the corresponding label. Note that it's a generator object and hence the loop will stop once
        # one image has been loaded and then return that image and the corresponding label. Next time the function is
        # called the next image and corresponding label will be returned and so forth.
        for k in xrange(len(paths_shuffled)):
            path = paths_shuffled[k]
            imz = np.zeros((3, 256, 256), dtype='float32')
            im = skimage.io.imread(path)
            if im.shape[0] > im.shape[1]:
                im = im.transpose(1, 0, 2)
                print "transposing!!!!!!!!1"
            offset = (im.shape[1] - im.shape[0]) // 2
            imz[:, offset:offset+im.shape[0], :] = im.transpose(2, 0, 1).astype('float32') / 255.0
            if np.isnan(np.sum(imz)):
                print "NAN at %d" % k
            if np.max(imz) > 1.0:
                print "too large at %d" % k
            if np.max(imz) < 0.0:
                print "too small at %d" % k
            if labels is not None:
                yield imz, labels_shuffled[k]
            else:
                yield imz
        
        if not repeat:
            break


def gen_chunks(image_gen, image_size, chunk_size=8192, labels=True):
    chunk = np.zeros((chunk_size, 3, image_size[0], image_size[1]), dtype='float32')     # Initialise arrays to hold images
    
    if labels:
        chunk_labels = np.zeros((chunk_size, 1), dtype='float32')       # Initialise array to hold labels

    offset = 0

    for sample in image_gen:
        if labels:
            im, label = sample
        else:
            im = sample

        chunk[offset] = im

        if labels:
            chunk_labels[offset] = label

        offset += 1

        if offset >= chunk_size:
            if labels:
                yield chunk, chunk_labels, offset
            else:
                yield chunk, offset

            chunk = np.zeros((chunk_size, 3, image_size[0], image_size[1]), dtype='float32')

            if labels:
                chunk_labels = np.zeros((chunk_size, 1), dtype='float32')

            offset = 0

    if offset > 0:
        if labels:
            yield chunk, chunk_labels, offset
        else:
            yield chunk, offset


def fast_warp(img, tf, output_shape=(256, 256), mode='constant'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf.params
    img_wf = np.empty((3, output_shape[0], output_shape[1]), dtype='float32')
    for k in xrange(3):
        img_wf[k] = skimage.transform._warps_cy._warp_fast(img[k], m, output_shape=output_shape, mode=mode)
    return img_wf


def build_center_uncenter_transform(image_size):
    center_shift = np.array(image_size) / 2. - 0.5
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0), flip=False, image_size=(256, 256)):
    tform_center, tform_uncenter = build_center_uncenter_transform(image_size)

    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)

    if flip:
        rotation += 180
        shear += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=rotation, shear=shear, translation=translation)
    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform


def random_augmentation_transform(zoom_range=(1.0, 1.0), rotation_range=(0, 0), shear_range=(0, 0), 
                                  translation_range=(0, 0), do_flip=False, image_size=(256, 256)):
    """
    The function creates the different transformation parameters (a function call is used in order to
    introduce randomnes into the parameters)
    """

    # random shift
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation
    rotation = np.random.uniform(*rotation_range)

    # random shear
    shear = np.random.uniform(*shear_range)

    # flip
    flip = do_flip and (np.random.randint(2) > 0) # flip half of the time

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range))

    return build_augmentation_transform(zoom, rotation, shear, translation, flip, image_size)


def augment_image(img, image_size, color_pertube, augmentation_params={}):

    # Pertub RBG pixel intensities
    if color_pertube:
        img = pca_color_pertub(img)

    # Create image transform parameters
    tform_augment = random_augmentation_transform(image_size=image_size, **augmentation_params)

    # Perform image transform and return the augmented image
    return fast_warp(img, tform_augment, output_shape=image_size).astype('float32')


def pca_color_pertub(im):
    """
    This functions adds multiples of the principal components of the covariance matrix
    for RBG pixel values. The intuition behind this distortion of the images is to make
    the object identity invariant to changes in the intenisty and color of the illumination.
    See Krizhevsky et. al 2012 for a more in depth explanation of the method.

    Note that it is assumed that the depth of the picture is the first dimension in image
    array i.e. image.shape = (depth, height or width, height or width).
    """

    # Calculate covariance matrix
    n = im.shape[1]*im.shape[2]
    A = im.reshape([im.shape[0], n])
    A_mean = np.mean(A, axis=1, keepdims = True)
    Q = A - A_mean
    cov = 1.0/(n-1) * np.dot(Q, Q.T)

    # Get eigenvalues and eigenvectors
    eigVal, eigVec = np.linalg.eig(cov)

    # Add multiples of the found principal components to each RBG pixel
    alpha = np.random.normal(0, 0.1, im.shape[0])
    color_distort = np.dot(eigVec, np.expand_dims(alpha*eigVal, axis=1))
    im_distorted = A + color_distort
    im_distorted = im_distorted.reshape(im.shape)

    return im_distorted