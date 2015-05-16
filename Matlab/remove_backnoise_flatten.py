# from scipy import ndimage
import skimage.feature
import os
import skimage.io
import skimage.color
import numpy as np
# import matplotlib.pyplot as plt
import sys
from clint.textui import progress
import glob
# import time
# from memory_profiler import profile as memory_profiling
import multiprocessing
import functools
import pathos.multiprocessing as mp
import PIL.Image
import getopt


def image_load(path):
    # print "load1"
    # print "path is:"
    # print path
    image = skimage.io.imread(path)
    # print "load2"
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
    # print "1"
    img = image_load(path_in)
    # print "2"
    img = remove_background_noise(img)
    # print "3"
    img = flatten_hist(img)
    # print "4"
    name = get_name_image(path_in)
    # print "5"
    image_save(path_out, name, img)


def main():

    # path_in = "/home/ubuntu/BIG_data/test_1024by1024_cropped"
    # path_out = "/home/ubuntu/BIG_data/test_1024_nonoise_flatten"

    path_in = "/home/ubuntu/BIG_data/train_1024by1024_cropped"
    path_out = "/home/ubuntu/BIG_data/train_1024_nonoise_flatten"

    # path_in = "/home/ubuntu/BIG_data/test_210by210_cropped"
    # path_out = "/home/ubuntu/BIG_data/test_210_nonoise_flatten"


    # path_in = "/home/ubuntu/BIG_data/train_210by210_cropped"
    # path_out = "/home/ubuntu/BIG_data/train_210_nonoise_flatten"

    if not os.path.exists(path_out):
        os.makedirs(path_out, mode=0755)
    image_names = glob.glob(os.path.join(path_in, "*.jpeg"))
    if (len(image_names) == 0):
        image_names = glob.glob(os.path.join(path_in, "*.png"))
    image_names.sort
    num_cores = multiprocessing.cpu_count()
    #p for parallel
    p = mp.Pool(num_cores)
    arg1 = image_names
    arg2 = len(arg1)*[path_out]
    arguments = zip(arg1, arg2)
    # p.map(lambda i: worker(i), (arguments))
    L = len(arguments)   

    batch_size =  num_cores
    batches =  L//batch_size

    
    for batch in progress.bar(xrange(batches)):
        if batch == (batches-1):
            arg = arguments[batch*batch_size:]
            # print arg
        else:
           start = batch * batch_size
           end = start + batch_size
           arg = arguments[start:end]

        #print get_name_image(arg[0][0])
        p.map(lambda i: worker(i), arg)
    # print "type %s and length %d" % (str(type(arg)), end-start )
        

    # arg = arguments[45:46]
    # p.map(lambda i: worker(i), arg)
    # print "what"
    # print "\n"*5

    # arg = arguments[46:47]
    # p.map(lambda i: worker(i), arg)



if __name__ == '__main__':
    main()