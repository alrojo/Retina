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
import getopt


def image_load(path):
    image = skimage.io.imread(path)
    return image


def remove_background_noise(image):
    """Watch out, this function transforms inplace"""
    h, w = image.shape[:2]
    ma = np.mean(image, axis=2)
    v = ma[np.rint(h/4):np.rint(3*h/4), np.rint(w/4):np.rint(3*w/4)]
    threshold = max(np.max(0.3*np.median(v)), 0.75*np.min(v))
    mask = (ma < threshold)
    image[mask] = 0
    return image


def flatten_hist(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    image = cdf[image]
    return image


def image_save(path, name, image):
    save_as = os.path.join(path, name)
    save_as = os.path.relpath(save_as)
    skimage.io.imsave(save_as, image)
    return 0


def get_name_image(path):
    name = path.split('/')[-1]
    return name


def worker(arg):
    path_in , path_out = arg
    img = image_load(path_in)
    img = remove_background_noise(img)
    img = flatten_hist(img)
    name = get_name_image(path_in)
    image_save(path_out, name, img)


def main():
    path_in = "/home/morten/Git_and_dropbox_not_friends/Retina/sample/resized_and_cropped"
    path_out = "/home/morten/Git_and_dropbox_not_friends/Retina/sample/resized_and_cropped/removed_back_ground"

    if not os.path.exists(path_out):
        os.makedirs(path_out, mode=0755)
    image_names = glob.glob(os.path.join(path_in, "*.jpeg"))
    if (len(image_names) == 0):
        image_names = glob.glob(os.path.join(path_in, "*.png"))
    image_names.sort
    num_cores = num_cores = multiprocessing.cpu_count()
    #p for parallel
    p = mp.Pool(num_cores)
    arg1 = image_names
    arg2 = len(arg1)*[path_out]
    arguments = zip(arg1, arg2)
    p.map(lambda i: worker(i), arguments)


if __name__ == '__main__':
    main()
