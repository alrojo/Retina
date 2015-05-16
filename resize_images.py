import os, sys
from PIL import Image
from clint.textui import progress
import glob

in_folder  = str(sys.argv[1])

size = 512, 512

in_paths = glob.glob("/Users/dawi/Desktop/kaggle/train/*.jpeg")

# replace pattern
pattern_from = "/train/"
pattern_to   = "/train_fixed_512/"

for infile in progress.bar(in_paths):
	im = Image.open(infile)
	im.thumbnail(size, Image.ANTIALIAS)
	outfile = infile.replace(pattern_from, pattern_to)
	outfile = outfile.replace("jpeg","png")
	im.save(outfile)
    # outfile = os.path.splitext(infile)[0] + ".thumbnail"
    # if infile != outfile:
    #     try:
    #         im = Image.open(infile)
    #         im.thumbnail(size, Image.ANTIALIAS)
    #         im.save(outfile, "JPEG")
    #     except IOError:
    #         print "cannot create thumbnail for '%s'" % infile