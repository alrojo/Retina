import glob 
import os

path = []
path.append("/home/ubuntu/BIG_data/test_1024by1024_cropped")
path.append("/home/ubuntu/BIG_data/train_1024by1024_cropped")
path.append("/home/ubuntu/BIG_data/test_210by210_cropped")
path.append("/home/ubuntu/BIG_data/train_210by210_cropped")
path.append("/home/ubuntu/BIG_data/test")
path.append("/home/ubuntu/BIG_data/train")
path.append("/home/ubuntu/BIG_data/test")
path.append("/home/ubuntu/BIG_data/train")

for path_in in path:
	print "For %s: " % path_in
	image_names = glob.glob(os.path.join(path_in, "*.jpeg"))
	if (len(image_names) == 0):
	    image_names = glob.glob(os.path.join(path_in, "*.png"))

	unique = set(image_names)

	print "There are %d unique files" % len(unique)
	print "and %d repeats" % (len(image_names)-len(unique))
	print "\n"*2

