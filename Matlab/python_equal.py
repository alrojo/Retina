import os 
import glob 
from joblib import Parallel, delayed  
import multiprocessing
import skimage.io
import numpy as np
import functools 
import pathos.multiprocessing as mp


# The important info I should bundle up and pass around as a dict

def test_fun1(x):
    b = x * 3
    return b, x

def test_fun2((a,b)):
    print a , b
    


def do_nothing_fun(input):
    return input

def image_load(path):
    image = skimage.io.imread(path)   
    return path, image

def image_save((path, image)):
    name = get_name_image(path)
    skimage.io.imsave(name, image)
    return 0


    # h,w = image.shape
 #    if size(A,1)>size(A,2) 
 #        A=rot90(A);
 #    end


def get_name_image(image):
    return image.split('/')[-1]


def name_to_name_and_path(list):
    """list should be a list of pathnames to the images"""
    paths = np.asarray(list).copy().reshape(len(list),1).shape
    names = np.vectorize(get_name_image)
    return np.concatenate((name,paths),axis=1)


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
    image_names = glob.glob(os.path.join(directory,"*.jpeg"))
    image_names.sort
    for i in image_names:
        print "globbed %s" % i

    # Doing it in parallel
    # Note that the input/outputs need to match 
    num_cores = multiprocessing.cpu_count()
    transform_image = transform_pipeline(image_load, image_save)
    results = Parallel(n_jobs=num_cores)(delayed(transform_image)(i) for i in images_names) 
    return 0 


def return_zero(input):
    print "length of inputs are %d" % len(input)
    print "and path is %s" % input[0]
    return input[-1]


#scripting 
#global development
development = True
image_out_id = "512_with_special_sauce"
directory = '/home/morten/Git_and_dropbox_not_friends/Retina/sample'
save_out = os.path.join(directory, image_out_id)
if not os.path.exists(save_out):
    os.makedirs(save_out, mode=0755)
image_names = glob.glob(os.path.join(directory,"*.jpeg"))
image_names.sort
for i in image_names:
    print "globbed %s" % i

# Doing it in parallel
# Note that the input/outputs need to match 
num_cores = multiprocessing.cpu_count()
transform_image = transform_pipeline(return_zero, do_nothing_fun, image_load)
blah = transform_pipeline(test_fun2, test_fun1)

# test
path_test = image_names[0]
test = transform_image(path_test)
# assert(test[-1]==0)

# So pickle sucks, and that's why parallel doesn't work
# Parallel(n_jobs=2)(delayed(transform_image)(i) for i in image_names) 

#
p = mp.Pool(2)
p.map(lambda i: transform_image(i), image_names)
# return 0 


# TODO: see http://matthewrocklin.com/blog/work/2013/12/05/Parallelism-and-Serialization/
# TODO: learn pathos ...see imported modules
# install from here instead https://github.com/uqfoundation/pathos

# if __name__ == '__main__':f
#     program = main()
#     if program==0:
#         print "finished"


# isntall 
    


    
    

    



	# info = {}
	# for i in image_names:
	# 	key = i.split('sample/')[-1]
	# 	info[key] = []

    # transform_image(f)
    




    
	
    




    #fun.usage()
    #exit()

    # f = {"1": image_load, "2":do_nothing_fun, "3":do_nothing_fun, 
    #     	 "4":do_nothing_fun, "5":do_nothing_fun,}
    # applied = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    # for i in applied:
    # 	if str(i) !=  do_nothing