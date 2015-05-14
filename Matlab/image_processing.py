# from scipy import ndimage
import skimage.feature
import os
import skimage.io
import skimage.color
import numpy as np
# import matplotlib.pyplot as plt
import sys
# from clint.textui import progress
import glob
# import time
# from memory_profiler import profile as memory_profiling
import multiprocessing
import functools
import pathos.multiprocessing as mp
import PIL.Image
# from joblib import Parallel, delayed # got ditched to pathos


def do_nothing_fun(input):
    return input


def image_load(path):
    image = skimage.io.imread(path)
    return path, image


def image_save((image, name, path)):
    save_as = os.path.join(path, name)
    save_as = os.path.relpath(save_as)
    skimage.io.imsave(save_as, image)
    return 0


def get_name_image(image_path):
    name = image_path.split('/')[-1]
    return name


def flatten_hist(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img = cdf[img]
    return img


def remove_background_noise(image):
    """Watch out, this function transforms inplace"""
    h, w = image.shape[:2]
    ma = np.mean(image, axis=2)
    v = ma[np.rint(h/4):np.rint(3*h/4), np.rint(w/4):np.rint(3*w/4)]
    threshold = max(np.max(0.3*np.median(v)), 0.75*np.min(v))
    mask = (ma < threshold)
    image[mask] = 0
    return image


def convert_to_uin8(image):
    return image.astype(np.uint8)


def rotate_90_degrees(image, direction='clockwise'):
    try:
        assert(direction == 'clockwise' or direction == 'counterclockwise')
    except AssertionError:
        print ('direction argument to "roate_90_degrees" must'
               'be "clockwise" or "counterclockwise"')
        sys.exit()
    # test if greyscale
    if len(image.shape) < 3:
        image = image.T
    else:
        image = np.transpose(image, (1, 0, 2))

    if direction == 'counterclockwise':
        image = np.flipud(image)

    return image


def normalize(img):
    """Not quite normalizing but rather making sure numbers
    are in teh interval 0-255"""
    min = np.amin(img)
    img = img-min
    max = np.amax(img)
    img = img/float(max)
    img = img*255
    img = img.astype(np.uint8)
    return img


def rgb_to_yiq(img):
    img = normalize(img)
    img[:, :, 0] = img[:, :, 0]/255.0
    img[:, :, 1] = (img[:, :, 1]-255/2.0)*0.5957
    img[:, :, 2] = (img[:, :, 2]-255/2.0)*0.5229
    A = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321], [0.212, -0.528, 0.311]])
    h, w = img.shape[:2]
    for i in xrange(h):
        for j in xrange(w):
            img[i, j, :] = np.dot(img[i, j, :], A)

    img = img.astype(np.uint8)
    return img


def from_rgb_to_hsv(im):
    return skimage.color.convert_colorspace(im, 'RGB', 'HSV')


def pil_resize(image_path, image_save_path, size):
    name = get_name_image(image_path)
    im = PIL.Image.open(image_path)
    im.thumbnail((size, size), PIL.Image.LANCZOS)
    d = {key: value for (key, value) in zip(['w', 'h'], im.size)}
    dim_max = max(d, key=d.get)
    dim_min = min(d, key=d.get)
    dif = size - d[dim_min]
    c = dif//2
    d[dim_min] = d[dim_min]+dif % 2
    d[dim_max] = [0+c, d[dim_max]-c]
    d[dim_min] = [0, d[dim_min]]
    lower, upper, left, right = [element for tpl in d.values() for element in tpl]
    box = (left, lower, right, upper)
    im = im.crop(box)
    name = name.replace("jpeg", "png")
    outfile = os.path.join(image_save_path, name)
    im.save(outfile)


# These functions allow for a pipelining of the above
# functions and for parallelization
def transform_pipeline(*functions):
    """Functional programming ;)"""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)


def main():
    global development
    development = True
    image_out_id = "512_with_special_sauce"
    directory = '/home/morten/Git_and_dropbox_not_friends/Retina/sample'
    save_out = os.path.join(directory, image_out_id)
    if not os.path.exists(save_out):
        os.makedirs(save_out, mode=0755)
    image_names = glob.glob(os.path.join(directory, "*.jpeg"))
    if (len(image_names) == 0):
        image_names = glob.glob(os.path.join(directory, "*.png"))
    image_names.sort

    num_cores = multiprocessing.cpu_count()
    transform_image = transform_pipeline(image_load, image_save)
    p = mp.Pool(num_cores)
    p.map(lambda i: transform_image(i), image_names)

    return 0


# Misc. function I might not use
def name_to_name_and_path(list):
    """list should be a list of pathnames to the images"""
    paths = np.asarray(list).copy().reshape(len(list), 1).shape
    names = np.vectorize(get_name_image)
    return np.concatenate((names, paths), axis=1)


def return_zero(input):
    print "length of inputs are %d" % len(input)
    print "and path is %s" % input[0]
    return input[-1]


if __name__ == '__main__':
    program = main()


####################################################


# from PIL import Image

notes = """
NOTES:
    - The important info I should bundle up and pass around as a dict
    - I should make a function that can run through all the images and make sure they
        - One function to make sure they are not rotated.
        - One function to make sure that after resizing they all have the same dimensions
    - add arguments and usage()
# TODO: see http://matthewrocklin.com/blog/work/2013/12/05/Parallelism-and-Serialization/
# TODO: learn pathos ...see imported modules
# install from here instead https://github.com/uqfoundation/pathos
"""
