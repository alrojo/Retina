import numpy as np 
import data
import glob
import os
import pandas as pd
import skimage.io


if os.path.exists("data/images_train.npy.gz"):
    print "generating train dataset file"
    csv = pd.read_csv("data/trainLabels.csv")
    filenames_train = np.array(csv['image'])

    num_images = len(filenames_train)

    mean1_train = 0
    mean2_train = 0
    mean3_train = 0

    std1_train = 0
    std2_train = 0
    std3_train = 0

    l = 0

    images = np.zeros((num_images, 3, 96, 96), dtype='float32')
    for k, fn in enumerate(filenames_train):
        if k % 100 == 0:
            print k

        im = skimage.io.imread(os.path.join("/home/josca/davidData/Retina/David/train_256", "%s.png" % fn))
        a = 0

        # Calculate mean and std. for RBG values
        mean1_train += im[:, :, 0].mean()
        mean2_train += im[:, :, 1].mean()
        mean3_train += im[:, :, 2].mean()

        std1_train += im[:, :, 0].std()
        std2_train += im[:, :, 1].std()
        std3_train += im[:, :, 2].std()

        l += 1

    mean1_train /= l
    mean2_train /= l
    mean3_train /= l

    std1_train /= l
    std2_train /= l
    std3_train /= l

    meanAndStd = [mean1_train, mean2_train, mean3_train, std1_train, std2_train, std3_train]

    np.save('data/meanAndStd_256.npy', meanAndStd)


 #       offset = (96 - im.shape[0]) // 2
 #       images[k, :, offset:offset+im.shape[0], :] = im.transpose(2, 0, 1).astype('float32') / 255.0

    # Save training images


#    np.save('data/images_train.npy', images)
#    os.system('gzip data/images_train.npy')
"""
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

    # Save test images
    np.save('data/images_test.npy', images)
    os.system('gzip data/images_test.npy')



"""
