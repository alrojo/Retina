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
    
    return size, rotated, name





path_to_images = "/home/ubuntu/BIG_data/test"
image_names = glob.glob(os.path.join(path_to_images, "*.jpeg"))
if (len(image_names) == 0):
    image_names = glob.glob(os.path.join(path_to_images, "*.png"))

image_names.sort

SIZE = 5000
for i in progress.bar(image_names):
    size, rotated, name = get_info(i)
    if rotated:
        print "%s is rotated" % name
    if size < SIZE:
        SIZE = size
        small = name

print "minimum size is: %d" % SIZE

try: 
    small
    print "minimimum size was for image: %s" % small
except NameError:
    print "Found no minimum sized image"



 # minimum image for raw training set. 15942_left.jpeg  433 by 289 pixels
 # rsync -azP ubuntu@52.1.254.11:/home/ubuntu/BIG_data/train/15942_left.jpeg .


 # minimum image for raw testing set.  769_left.jpeg  320 by 211 pixels
 # rsync -azP ubuntu@52.1.254.11:/home/ubuntu/BIG_data/test/769_left.jpeg .

# and there is a rotated image in the test set: 15840_left.jpeg
# rsync -azP ubuntu@52.1.254.11:/home/ubuntu/BIG_data/test/15840_left.jpeg .

# I should make a program that deletes every image smaller than HD


