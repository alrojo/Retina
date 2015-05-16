import glob 
import os
import PIL.Image
from clint.textui import progress
import numpy as np
import time

path = []
# path.append("/home/ubuntu/BIG_data/test_1024by1024_cropped")
# path.append("/home/ubuntu/BIG_data/train_1024by1024_cropped")
# path.append("/home/ubuntu/BIG_data/test_210by210_cropped")
path.append("/home/ubuntu/BIG_data/train_210by210_cropped")
# path.append("/home/ubuntu/BIG_data/test")
# path.append("/home/ubuntu/BIG_data/train")
# path.append("/home/ubuntu/BIG_data/test_210_nonoise_flatten")
# path.append("/home/ubuntu/BIG_data/train_210_nonoise_flatten")
output = []

for path_in in path:
    start_time = time.time()
    name = path_in.split("/")[-1]
    save_out = os.path.join(os.getcwd(),name+".npy")
    output = []
    print "For %s: " % path_in
    image_names = glob.glob(os.path.join(path_in, "*.jpeg"))
    if (len(image_names) == 0):
        image_names = glob.glob(os.path.join(path_in, "*.png"))

    for image_path in progress.bar(image_names):
       name_image = image_path.split("/")[-1]   
       im = PIL.Image.open(image_path)
       w,h = im.size
       hist = np.asarray(im.histogram(), dtype=np.float)
       factor = np.arange(256)
       divisor = w * h
       r = (hist[0:256]*factor).sum()/divisor 
       g = (hist[256:512]*factor).sum()/divisor 
       b = (hist[512:768]*factor).sum()/divisor 
       
       output.append([name_image, r, g, b])

    output = np.asarray(output)
    np.save(save_out,output)
    print path_in + "took " + str(time.time() - start_time) + " s" 
    

# # rsync -azP ubuntu@52.7.33.169:/home/ubuntu/BIG_data/Retina/Matlab/train_210by210_cropped.npy