import numpy as np 
import data
import glob
import os
import pandas as pd
import skimage.io


if not os.path.exists("data/images_train.npy.gz"):
    print "generating train dataset file"
    csv = pd.read_csv("data/trainLabels.csv")
    filenames_train = np.array(csv['image'])

    num_images = len(filenames_train)

    images = np.zeros((num_images, 3, 96, 96), dtype='float32')
    for k, fn in enumerate(filenames_train):
        if k % 100 == 0:
            print k

        im = skimage.io.imread(os.path.join("data/train", "%s.jpeg" % fn))
        offset = (96 - im.shape[0]) // 2
        images[k, :, offset:offset+im.shape[0], :] = im.transpose(2, 0, 1).astype('float32') / 255.0

    data.save_gz('data/images_train.npy.gz', images)


if not os.path.exists("data/images_test.npy.gz"):
    print "generating test dataset file"
    paths_test = glob.glob("data/test/*.jpeg")
    paths_test.sort()

    num_images = len(paths_test)

    images = np.zeros((num_images, 3, 96, 96), dtype='float32')

    for k, path in enumerate(paths_test):
        if k % 100 == 0:
            print k

        im = skimage.io.imread(path)

        if im.shape[0] > im.shape[1]:
            im = im.transpose(1, 0, 2)
            print "transposing!!!!!!!!!1"

        offset = (96 - im.shape[0]) // 2
        images[k, :, offset:offset+im.shape[0], :] = im.transpose(2, 0, 1).astype('float32') / 255.0

    data.save_gz('data/images_test.npy.gz', images)




