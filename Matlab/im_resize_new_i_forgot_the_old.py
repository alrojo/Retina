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
# from memory_profiler import profile as memory_profiling



def get_name_image(path):
    name = path.split('/')[-1]
    return name


def pil_resize(image_path):
    global size
    global path_out
    image_save_path = path_out
    name = get_name_image(image_path)
    im = PIL.Image.open(image_path)
    # to get everyone the same size
    # size_standard = 212
    # im.thumbnail(size_standard, PIL.Image.LANCZOS)
    d = {key: value for (key, value) in zip(['w', 'h'], im.size)}
    # print "1: " + str(d )
    dim_max = max(d, key=d.get)
    # print "2: " + str(dim_max)
    dim_min = min(d, key=d.get)
    if dim_max == dim_min:
        dim_max = 'w'
        dim_min = 'h'
    # print "3: " + str(dim_min)
    dif = d[dim_max] - d[dim_min]
    # print "4: " + str(dif)

    c = dif//2
    # print "5: " + str(c)

    d[dim_min] = d[dim_min]+dif % 2
    d[dim_max] = [0+c, d[dim_max]-c]
    d[dim_min] = [0, d[dim_min]]
    # print d
    lower, upper, left, right = [element for tpl in d.values() for element in tpl]
    # print "1\n"
    box = (left, lower, right, upper)
    # print "2\n"
    im = im.crop(box)
    # print "3\n"
    im.thumbnail((size,size) , PIL.Image.LANCZOS)
    # print "4\n"
    name = name.replace("jpeg", "png")
    # print "5\n"
    outfile = os.path.join(image_save_path, name)
    im.save(outfile)
    return 0


def worker(path_in):
    pil_resize(path_in)
    return 0
    
# @memory_profiling
def main():
    global size
    global path_out
    start_time = time.time()
    size = 210
    path_in = "/home/ubuntu/BIG_data/train"
    path_out = "/home/ubuntu/BIG_data/train_210by210_cropped"

    if not os.path.exists(path_out):
        os.makedirs(path_out, mode=0755)
    image_names = glob.glob(os.path.join(path_in, "*.jpeg"))
    if (len(image_names) == 0):
        image_names = glob.glob(os.path.join(path_in, "*.png"))
    image_names.sort
    num_cores = num_cores = multiprocessing.cpu_count()
    #p for parallel
    p = mp.Pool(num_cores)
    # print "Number of cores %d" % num_cores  # So gp2.x2large has 8 cores
    # arg1 = image_names
    # L = len(arg1)
    # arg2 = L*[path_out]    
    # arg3 = L*[size] 
    # arguments = zip(arg1, arg2, arg3)
    L = len(image_names)
    batch_size = 7
    batches = L//7

    for batch in progress.bar(xrange(batches)):
        if batch == (batches-1):
            arg = image_names[batch*batch_size:]
            # print arg
        else:
           start = batch * batch_size
           end = start + batch_size
           arg = image_names[start:end]
        # print "type %s and length %d" % (str(type(arg)), end-start )
        start = batch * batch_size
        end = start + batch_size
        arg = image_names[start:end]
        p.map(lambda i: worker(i), arg)
    
    print ("prgram completed in %f seconds") % (time.time()-start_time)

if __name__ == '__main__':
    main()

 
    # if batch == (batches-1):
    #     arg = image_names[batch*batch_size:]
    #     # print arg
    # else:
    #    start = batch * batch_size
    #    end = start + batch_size
    #    arg = image_names[start:end]
    #print "type %s and length %d" % (str(type(arg)), end-start )
    # batch = 2004
    # start = batch * batch_size
    # end = start + batch_size
    # arg = image_names[start:end]
    # for i in arg:
    #     print arg
    
    # print "\n"*5 + "batch 2005"+ 2*"\n"

    # batch = 2005
    # start = batch * batch_size
    # end = start + batch_size
    # arg = image_names[start:end]
    # for i in arg:
    #     print arg

    # print "\n"*10

    # for i in image_names[start:end]:
    #     print "\n"*3
    #     print i
       
    #     if (i ==  '/home/ubuntu/BIG_data/train/33869_left.jpeg'):
    #         pil_resize(i)    
    #     print 
       


    #     if (i == "/home/ubuntu/BIG_data/train/22321_left.jpeg"):
    #         pil_resize(i)    
    #     print 
        

    # p.map(lambda i: worker(i), arg)
    
    # print ("prgram completed in %f seconds") % (time.time()-start_time)




    # for batch in xrange(batches):
