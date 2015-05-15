# from scipy import ndimage
# import skimage.feature
import os
import skimage.io
import skimage.color
import numpy as np
# import matplotlib.pyplot as plt
import sys
from clint.textui import progress
import glob
import time
import multiprocessing
import functools
import pathos.multiprocessing as mp
import PIL.Image
import getopt
# from joblib import Parallel, delayed # got ditched to pathos
from memory_profiler import profile as memory_profiling



def get_name_image(path):
    name = path.split('/')[-1]
    return name


def pil_resize(image_path, image_save_path, size):
    name = get_name_image(image_path)
    im = PIL.Image.open(image_path)
    # to get everyone the same size
    size_standard = 1800
    im.thumbnail((size_standard, size_standard), PIL.Image.LANCZOS)
    d = {key: value for (key, value) in zip(['w', 'h'], im.size)}
    dim_max = max(d, key=d.get)
    dim_min = min(d, key=d.get)
    dif = size_standard - d[dim_min]
    c = dif//2
    d[dim_min] = d[dim_min]+dif % 2
    d[dim_max] = [0+c, d[dim_max]-c]
    d[dim_min] = [0, d[dim_min]]
    lower, upper, left, right = [element for tpl in d.values() for element in tpl]
    box = (left, lower, right, upper)
    im = im.crop(box)
    im.thumbnail((size, size), PIL.Image.LANCZOS)
    name = name.replace("jpeg", "png")
    outfile = os.path.join(image_save_path, name)
    im.save(outfile)
    return 0




def worker(arg):
    path_in , path_out, size = arg
    pil_resize(path_in, path_out, size)
    return 0
    
@memory_profiling
def main():

    start_time = time.time()
    size = 424
    path_in = "/home/morten/Git_and_dropbox_not_friends/Retina/sample/"
    path_out = "/home/morten/Git_and_dropbox_not_friends/Retina/sample/same_size2"

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
    L = len(arg1)
    arg2 = L*[path_out]    
    arg3 = L*[size] 
    arguments = zip(arg1, arg2, arg3)
    p.map(lambda i: worker(i), progress.bar(arguments))
    print ("prgram completed in %f seconds") % (time.time()-start_time)

if __name__ == '__main__':
    main()
