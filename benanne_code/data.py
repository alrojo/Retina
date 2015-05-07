import numpy as np 
import skimage.transform
import skimage.io
import pandas as pd
import os
import glob


split = np.load("data/split.pkl")

csv = pd.read_csv("data/trainLabels.csv")
filenames_train = np.array(csv['image'])
labels_train = np.array(csv['level'])
labels_my_train = labels_train[split['train']]
labels_my_valid = labels_train[split['valid']]

num_images = len(labels_train)

paths_train = np.array([os.path.join("train_new", "%s.jpeg" % filename) for filename in filenames_train])

paths_my_train = paths_train[split['train']]
paths_my_valid = paths_train[split['valid']]

paths_test = glob.glob("test_new/*.jpeg")
paths_test.sort()
paths_test = np.array(paths_test)


def gen_images(paths, labels=None, shuffle=False, repeat=False):
    paths_shuffled = np.array(paths)

    if labels is not None:
        labels_shuffled = np.array(labels)

    while True:
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(paths_shuffled)
            if labels is not None:
                np.random.set_state(state)
                np.random.shuffle(labels_shuffled)

        for k in xrange(len(paths_shuffled)):
            path = paths_shuffled[k]
            im = skimage.io.imread(os.path.join("data", path))
            im = im.transpose(2, 0, 1).astype('float32') / 255.0

            if labels is not None:
                yield im, labels_shuffled[k]
            else:
                yield im
        
        if not repeat:
            break


def gen_chunks(image_gen, chunk_size=8192, labels=True):
    chunk = np.zeros((chunk_size, 3, 96, 96), dtype='float32')
    
    if labels:
        chunk_labels = np.zeros((chunk_size, 1), dtype='float32')

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

            chunk = np.zeros((chunk_size, 3, 96, 96), dtype='float32')

            if labels:
                chunk_labels = np.zeros((chunk_size, 1), dtype='float32')

            offset = 0

    if offset > 0:
        if labels:
            yield chunk, chunk_labels, offset
        else:
            yield chunk, offset


def fast_warp(img, tf, output_shape=(96, 96), mode='constant'):
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


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0), flip=False, image_size=(96, 96)):
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
                                  translation_range=(0, 0), do_flip=False, image_size=(96, 96)):
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


def augment_image(img, augmentation_params={}):
    tform_augment = random_augmentation_transform(**augmentation_params)
    return fast_warp(img, tform_augment, output_shape=(96, 96)).astype('float32')
