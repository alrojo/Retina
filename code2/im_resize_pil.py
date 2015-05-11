import os, sys
from PIL import Image
from clint.textui import progress
import glob
import time 
from memory_profiler import profile as memory_profiling

start_time = time.time()
 

# @memory_profiling
# def mem():
#     for i in progress.bar(xrange(10)): 
#         fun()


def fun(): 
    #in_folder  = str(sys.argv[1])

    size = 512, 512
    in_paths = glob.glob("/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/sample/*.jpeg")                                  

    # replace pattern
    pattern_from = "/sample/"
    pattern_to   = "/pil_512_trans_jpg/"

    for i, infile in enumerate(in_paths):
        im = Image.open(infile)
        if i == 0:
            im.thumbnail(size, Image.LANCZOS)
            d = {key: value for (key, value) in zip(['w','h'],im.size)}
            dim_max = max(d,key=d.get)
            dim_min = min(d,key=d.get)
            c = (size[0] - d[dim_min])//2
            print "dim is %s" % str(im.size)
            print c
            #print d
            #print dim_max
            d[dim_max] = [0+c,d[dim_max]-c]
            d[dim_min] = [0,d[dim_min]]
            print d.values()
            left, right, lower, upper = [element for tpl in d.values() for element in tpl]
            print "left %d, right %d, lower %d, upper %d" % (left, right, lower, upper)
            "box = (left,upper,right,lower)"
            box = (c,0,427,341)
            print "crop box is %s" % str(box)
            size2 = max(right-left, upper-lower)   
            print size2    
        im.thumbnail(size, Image.LANCZOS)
        print "size is %s" % str(size)
        im = im.crop(box)
        outfile = infile.replace(pattern_from, pattern_to)
    	#outfile = outfile.replace("jpeg","png")
    	
        im.save(outfile)
        # outfile = os.path.splitext(infile)[0] + ".thumbnail"
        # if infile != outfile:
        #     try:
        #         im = Image.open(infile)
        #         im.thumbnail(size, Image.ANTIALIAS)
        #         im.save(outfile, "JPEG")
        #     except IOError:
        #         print "cannot create thumbnail for '%s'" % infile





if __name__ == '__main__':
    #mem()
    fun()



print ("im_resize_pil.py took %f seconds") % (time.time()-start_time)