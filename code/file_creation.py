import numpy as np 
from fun import load
from os import system
from os.path import join
from fun import load, save


# Saving train labels
save(Paths.data_dir, {'train_labels': data.train_labels})



print "Loading train images"
train_images = load(Paths.X_train)

#Saving training images
save(Paths.data_dir, {'train_images': train_images})

#No more use for them 
del train_images

print "loading test images"
test_images = load('Paths.X_test')

#Saving test images 
save(Paths.data_dir, {'test_images': test_images})
del test_images

