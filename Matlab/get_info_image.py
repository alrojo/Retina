import sys
import glob 
import os
import PIL.Image
from clint.textui import progress 




def get_name_image(image_path):
    name = image_path.split('/')[-1]
    return name


def get_info(image_path):
    
    name = get_name_image(image_path)
    im = PIL.Image.open(image_path)
    w, h = im.size
    rotated = False
    if w < h:
        rotated = True
    size = min(w,h)
    
    return size, rotated





path_to_images = "/home/morten/Git_and_dropbox_not_friends/Retina/sample/resized_and_cropped"
image_names = glob.glob(os.path.join(path_to_images, "*.jpeg"))
if (len(image_names) == 0):
    image_names = glob.glob(os.path.join(path_to_images, "*.png"))

image_names.sort

SIZE = 5000
for i in progress.bar(image_names):
    size, rotated = get_info(i)
    if rotated:
        n = get_name_image(i)
        print "%s is rotated" % n

    if size < SIZE:
        SIZE = size

print "minimum size is: %d" % SIZE



